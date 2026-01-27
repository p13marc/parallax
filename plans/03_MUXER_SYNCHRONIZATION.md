# Plan 03: Muxer Synchronization

**Priority:** High (Short-term)  
**Effort:** Large (1-2 weeks)  
**Dependencies:** Plan 01 (Custom Metadata API), Plan 02 (Codec Wrappers)  
**Addresses:** Pain Point 1.3 (Muxer Element Model Mismatch)

---

## Problem Statement

The current element model assumes 1-input-1-output for transforms. Muxers fundamentally need:
- **N inputs** (video, audio, metadata streams)
- **1 output** (multiplexed container)
- **Synchronized timing** (output when all inputs have data for a given PTS window)

Currently, `TsMux` is a standalone struct requiring manual `write_pes()` calls with manual timestamp synchronization. It cannot be used as a pipeline element.

---

## Proposed Solution

Implement a proper `Muxer` trait and executor support for multi-input elements with PTS-based synchronization.

### High-Level Design

```rust
// In pipeline construction:
let mux = pipeline.add_node("tsmux", TsMuxElement::new(config));

// Multiple inputs linked to same muxer
pipeline.link_pads(video_encoder, "src", mux, "video_0")?;
pipeline.link_pads(audio_encoder, "src", mux, "audio_0")?;
pipeline.link_pads(klv_source, "src", mux, "data_0")?;

// Executor handles synchronization automatically
pipeline.run().await?;
```

---

## Design

### Muxer Trait

```rust
/// Input from a specific pad with timestamp
pub struct MuxerInput {
    /// Which input pad this came from
    pub pad_id: PadId,
    /// The buffer data
    pub buffer: Buffer,
}

/// Output from muxer (usually container packets)
pub struct MuxerOutput {
    /// Multiplexed data
    pub data: Vec<u8>,
    /// Timestamp of this output segment
    pub pts: ClockTime,
}

/// Trait for N-to-1 multiplexer elements.
pub trait Muxer: Send {
    /// Called when a buffer arrives on any input pad.
    /// The muxer queues the buffer internally.
    fn push(&mut self, input: MuxerInput) -> Result<()>;
    
    /// Check if muxer is ready to produce output.
    /// Returns true if all required inputs have data up to the target PTS.
    fn can_output(&self) -> bool;
    
    /// Produce the next chunk of multiplexed output.
    /// Called by executor when can_output() returns true.
    fn pull(&mut self) -> Result<Option<MuxerOutput>>;
    
    /// Flush any remaining data (called at EOS).
    fn flush(&mut self) -> Result<Vec<MuxerOutput>>;
    
    /// Get the list of input pads this muxer expects.
    fn input_pads(&self) -> Vec<PadInfo>;
    
    /// Dynamically add an input pad (optional, for dynamic muxing).
    fn add_input_pad(&mut self, info: PadInfo) -> Result<PadId> {
        Err(Error::NotSupported("dynamic pads not supported".into()))
    }
    
    /// Get muxer name for debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Information about a muxer input pad
#[derive(Debug, Clone)]
pub struct PadInfo {
    pub name: String,
    pub stream_type: StreamType,
    pub required: bool,  // Must have data before outputting
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Video,
    Audio,
    Subtitle,
    Data,  // KLV, SCTE-35, etc.
}
```

### Synchronization Strategy

The muxer must decide when to output based on input timestamps:

```rust
pub struct MuxerSyncState {
    /// Target PTS for next output
    target_pts: ClockTime,
    /// Output interval (e.g., 40ms for 25fps)
    interval: ClockTime,
    /// Per-pad state
    pad_states: HashMap<PadId, PadState>,
}

struct PadState {
    /// Buffered data for this pad
    queue: VecDeque<TimestampedBuffer>,
    /// Last PTS seen on this pad
    last_pts: ClockTime,
    /// Whether this pad is required for output
    required: bool,
    /// EOS received on this pad
    eos: bool,
}

impl MuxerSyncState {
    /// Check if all required pads have data up to target PTS
    fn ready_to_output(&self) -> bool {
        for (_, state) in &self.pad_states {
            if state.required && !state.eos {
                // Need data at or past target PTS
                if state.last_pts < self.target_pts && state.queue.is_empty() {
                    return false;
                }
            }
        }
        true
    }
    
    /// Get all buffers with PTS <= target
    fn collect_for_output(&mut self) -> Vec<MuxerInput> {
        let mut inputs = Vec::new();
        for (pad_id, state) in &mut self.pad_states {
            while let Some(buf) = state.queue.front() {
                if buf.pts <= self.target_pts {
                    inputs.push(MuxerInput {
                        pad_id: *pad_id,
                        buffer: state.queue.pop_front().unwrap().buffer,
                    });
                } else {
                    break;
                }
            }
        }
        inputs
    }
    
    /// Advance target PTS
    fn advance(&mut self) {
        self.target_pts = self.target_pts + self.interval;
    }
}
```

### TsMuxElement Implementation

```rust
pub struct TsMuxElement {
    /// Inner mux logic
    inner: TsMux,
    /// Synchronization state
    sync: MuxerSyncState,
    /// Pad ID to PID mapping
    pad_to_pid: HashMap<PadId, u16>,
}

impl TsMuxElement {
    pub fn new(config: TsMuxConfig) -> Self {
        let mut sync = MuxerSyncState::new(ClockTime::from_millis(40));
        let mut pad_to_pid = HashMap::new();
        
        // Create pads from config tracks
        for track in &config.tracks {
            let pad_id = sync.add_pad(PadInfo {
                name: format!("{}_{}", track.stream_type.name(), track.pid),
                stream_type: track.stream_type.into(),
                required: track.stream_type.is_video(),  // Video required
            });
            pad_to_pid.insert(pad_id, track.pid);
        }
        
        Self {
            inner: TsMux::new(config),
            sync,
            pad_to_pid,
        }
    }
}

impl Muxer for TsMuxElement {
    fn push(&mut self, input: MuxerInput) -> Result<()> {
        self.sync.push(input.pad_id, input.buffer)?;
        Ok(())
    }
    
    fn can_output(&self) -> bool {
        self.sync.ready_to_output()
    }
    
    fn pull(&mut self) -> Result<Option<MuxerOutput>> {
        if !self.can_output() {
            return Ok(None);
        }
        
        let inputs = self.sync.collect_for_output();
        let mut output_data = Vec::new();
        
        // Write PSI tables periodically
        if self.should_write_psi() {
            output_data.extend(self.inner.write_psi());
        }
        
        // Write each input as PES
        for input in inputs {
            let pid = self.pad_to_pid[&input.pad_id];
            let pts = input.buffer.metadata().timestamp;
            let data = input.buffer.as_bytes();
            
            let pes_data = self.inner.write_pes(pid, data, Some(pts), None)?;
            output_data.extend(pes_data);
        }
        
        let pts = self.sync.target_pts;
        self.sync.advance();
        
        Ok(Some(MuxerOutput { data: output_data, pts }))
    }
    
    fn flush(&mut self) -> Result<Vec<MuxerOutput>> {
        // Process all remaining buffered data
        let mut outputs = Vec::new();
        while self.sync.has_buffered_data() {
            if let Some(output) = self.pull()? {
                outputs.push(output);
            } else {
                break;
            }
        }
        Ok(outputs)
    }
    
    fn input_pads(&self) -> Vec<PadInfo> {
        self.sync.pad_infos()
    }
}
```

### Executor Integration

The executor needs special handling for muxer nodes:

```rust
// In unified_executor.rs

async fn run_muxer_node(
    muxer: &mut dyn Muxer,
    input_channels: Vec<Receiver<Buffer>>,
    output_channel: Sender<Buffer>,
) -> Result<()> {
    let pad_ids: Vec<PadId> = muxer.input_pads().iter()
        .map(|p| p.id)
        .collect();
    
    loop {
        // Wait for input on any channel
        tokio::select! {
            // Try to receive from each input channel
            result = receive_any(&input_channels, &pad_ids) => {
                match result {
                    Some((pad_id, buffer)) => {
                        muxer.push(MuxerInput { pad_id, buffer })?;
                        
                        // Check if ready to output
                        while muxer.can_output() {
                            if let Some(output) = muxer.pull()? {
                                let buffer = output_to_buffer(output)?;
                                output_channel.send(buffer).await?;
                            }
                        }
                    }
                    None => {
                        // All inputs closed, flush
                        for output in muxer.flush()? {
                            let buffer = output_to_buffer(output)?;
                            output_channel.send(buffer).await?;
                        }
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// Receive from any of the channels, returning which one
async fn receive_any(
    channels: &[Receiver<Buffer>],
    pad_ids: &[PadId],
) -> Option<(PadId, Buffer)> {
    // Use tokio::select! macro generated for N channels
    // Or use a FuturesUnordered approach
    todo!()
}
```

### MuxerAdapter

Similar to other adapters, wrap Muxer for AsyncElementDyn:

```rust
pub struct MuxerAdapter<M: Muxer> {
    inner: M,
}

impl<M: Muxer + 'static> AsyncElementDyn for MuxerAdapter<M> {
    fn element_type(&self) -> ElementType {
        ElementType::Muxer
    }
    
    // Special handling needed - executor must detect Muxer type
    // and use run_muxer_node instead of standard process loop
}
```

---

## Implementation Steps

### Step 1: Define Muxer Trait and Types

**File:** `src/element/traits.rs` (or new `src/element/muxer.rs`)

- Add `Muxer` trait
- Add `MuxerInput`, `MuxerOutput`, `PadInfo`, `StreamType`
- Add `MuxerAdapter`

### Step 2: Implement MuxerSyncState

**File:** `src/element/muxer_sync.rs`

- Implement `MuxerSyncState` with PTS-based synchronization
- Per-pad queuing and EOS handling
- Configurable sync strategy (strict vs. loose)

### Step 3: Create TsMuxElement

**File:** `src/elements/mux/ts_element.rs`

- Wrap existing `TsMux` with `Muxer` trait
- Map pads to PIDs
- Implement push/pull with sync state

### Step 4: Update Executor

**File:** `src/pipeline/unified_executor.rs`

- Detect `ElementType::Muxer` nodes
- Use specialized `run_muxer_node` function
- Handle multiple input channels per node

### Step 5: Update Pipeline Graph

**File:** `src/pipeline/graph.rs`

- Allow multiple incoming edges to Muxer nodes
- Validate pad names match muxer's declared pads

### Step 6: Create Example

**File:** `examples/33_muxer_element.rs`

```rust
let mut pipeline = Pipeline::new();

// Video path
let video_src = pipeline.add_node("vsrc", video_test_src);
let encoder = pipeline.add_node("enc", encoder_element);
pipeline.link(video_src, encoder)?;

// Metadata path  
let klv_src = pipeline.add_node("klv", klv_source);

// Muxer
let mux = pipeline.add_node("mux", TsMuxElement::new(config));
pipeline.link_pads(encoder, "src", mux, "video_0")?;
pipeline.link_pads(klv_src, "src", mux, "data_0")?;

// Output
let sink = pipeline.add_node("sink", file_sink);
pipeline.link(mux, sink)?;

pipeline.run().await?;
```

---

## Synchronization Modes

Different use cases need different sync strategies:

### Strict Sync (Default)

Wait for data from all required pads before outputting:

```rust
fn ready_to_output(&self) -> bool {
    self.pad_states.iter()
        .filter(|(_, s)| s.required && !s.eos)
        .all(|(_, s)| s.last_pts >= self.target_pts || !s.queue.is_empty())
}
```

### Loose Sync (Low Latency)

Output as soon as video is available, include other streams if present:

```rust
fn ready_to_output(&self) -> bool {
    // Only wait for video
    self.pad_states.iter()
        .filter(|(_, s)| s.stream_type == StreamType::Video)
        .any(|(_, s)| !s.queue.is_empty())
}
```

### Timed Sync (Fixed Interval)

Output at fixed intervals, include whatever data is available:

```rust
fn ready_to_output(&self) -> bool {
    // Check if interval has elapsed since last output
    self.last_output_time.elapsed() >= self.interval
}
```

---

## Validation Criteria

- [ ] `Muxer` trait defined with push/pull model
- [ ] `MuxerSyncState` handles PTS-based synchronization
- [ ] `TsMuxElement` wraps existing `TsMux`
- [ ] Executor handles multi-input muxer nodes
- [ ] Pipeline allows multiple links to muxer
- [ ] Example 33 demonstrates video + metadata muxing
- [ ] Strict/loose/timed sync modes work
- [ ] EOS flushes all remaining data
- [ ] All existing tests pass

---

## Edge Cases

1. **Sparse streams:** Metadata at 10Hz, video at 25Hz
   - Solution: Don't require data for every output interval

2. **Out-of-order arrival:** Packets arrive before their PTS order
   - Solution: Buffer and sort by PTS before output

3. **One stream ends early:** Audio ends before video
   - Solution: Mark pad as EOS, continue with remaining streams

4. **Timestamp discontinuity:** Stream restarts with new base time
   - Solution: Detect and handle segment events

---

## Future Enhancements

1. **Dynamic pad creation:** Add streams during runtime
2. **Interleaving control:** Configure A/V interleave pattern
3. **Output buffering:** Configurable output buffer size
4. **QoS feedback:** Signal upstream when falling behind
5. **Multiple output formats:** Same muxer, different containers

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/element/muxer.rs` | New: Muxer trait, types |
| `src/element/muxer_sync.rs` | New: synchronization logic |
| `src/element/mod.rs` | Export new types |
| `src/elements/mux/ts_element.rs` | New: TsMuxElement |
| `src/elements/mux/mod.rs` | Export TsMuxElement |
| `src/pipeline/unified_executor.rs` | Multi-input muxer handling |
| `src/pipeline/graph.rs` | Multi-link to muxer validation |
| `examples/33_muxer_element.rs` | New example |
