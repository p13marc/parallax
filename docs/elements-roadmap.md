# Elements Roadmap

This document tracks the implementation status of pipeline elements, **ordered by value and complexity** to guide incremental implementation.

## Legend

- [x] Implemented
- [ ] Not yet implemented

**Complexity**: Low (L), Medium (M), High (H)
**Value**: Essential (E), High (H), Medium (M), Low (L)

---

## Priority Implementation Order

Elements are grouped into implementation tiers, ordered by **highest value first**, then **lowest complexity first** within each tier.

---

## Tier 1: High Value, Low Complexity (Quick Wins)

These elements provide immediate utility with minimal implementation effort.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [x] **PassThrough** | Transform | L | Identity element |
| [x] **NullSink** | Sink | L | Discard all buffers |
| [x] **NullSource** | Source | L | Produce empty buffers |
| [x] **Valve** | Transform | L | On/off flow control |
| [x] **Tee** | Routing | L | 1-to-N fanout |
| [x] **DataSrc** | Source | L | Inline data source |
| [x] **ConsoleSink** | Sink | L | Debug output |
| [ ] **Identity** | Transform | L | Pass-through with callbacks |
| [ ] **MemorySrc** | Source | L | Read from memory slice |
| [ ] **MemorySink** | Sink | L | Write to memory buffer |
| [ ] **Delay** | Timing | L | Fixed delay element |
| [ ] **SequenceNumber** | Metadata | L | Add sequence numbers |
| [ ] **Timestamper** | Metadata | L | Add/modify timestamps |
| [ ] **MetadataInject** | Metadata | L | Add/modify metadata |
| [ ] **BufferTrim** | Buffer | L | Trim to max size |
| [ ] **BufferSlice** | Buffer | L | Extract slice |

---

## Tier 2: High Value, Low-Medium Complexity (Core Functionality)

Essential elements that form the backbone of most pipelines.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [x] **FileSrc** | Source | L | Read from file |
| [x] **FileSink** | Sink | L | Write to file |
| [x] **AppSrc** | Source | M | Inject from application |
| [x] **AppSink** | Sink | M | Extract to application |
| [x] **Queue** | Transform | M | Async buffering with backpressure |
| [x] **RateLimiter** | Transform | L | Throughput limiting |
| [x] **Funnel** | Routing | M | N-to-1 merge |
| [x] **Concat** | Routing | M | Sequential concatenation |
| [x] **InputSelector** | Routing | M | N-to-1 switching |
| [x] **OutputSelector** | Routing | M | 1-to-N routing |
| [x] **TestSrc** | Source | L | Test pattern generator |
| [x] **FdSrc** | Source | L | Raw FD source |
| [x] **FdSink** | Sink | L | Raw FD sink |
| [ ] **Filter** | Transform | L | Generic predicate filter |
| [ ] **Map** | Transform | L | Apply function to contents |
| [ ] **Batch** | Transform | M | Combine N buffers into one |
| [ ] **Unbatch** | Transform | M | Split one buffer into N |
| [ ] **Chunk** | Transform | L | Fixed-size chunking |
| [ ] **SampleFilter** | Filter | L | Every Nth / random % |
| [ ] **Timeout** | Timing | M | Timeout with fallback |
| [ ] **Debounce** | Timing | M | Debounce rapid buffers |

---

## Tier 3: High Value, Medium Complexity (Network I/O)

Network elements are essential for distributed pipelines.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [x] **TcpSrc** | Network | M | TCP client/server source |
| [x] **TcpSink** | Network | M | TCP sink |
| [x] **AsyncTcpSrc** | Network | M | Async TCP source |
| [x] **AsyncTcpSink** | Network | M | Async TCP sink |
| [x] **UdpSrc** | Network | M | UDP receiver |
| [x] **UdpSink** | Network | M | UDP sender |
| [x] **AsyncUdpSrc** | Network | M | Async UDP source |
| [x] **AsyncUdpSink** | Network | M | Async UDP sink |
| [ ] **UnixSrc** | Network | L | Unix domain socket source |
| [ ] **UnixSink** | Network | L | Unix domain socket sink |
| [ ] **UdpMulticastSrc** | Network | M | Multicast receiver |
| [ ] **UdpMulticastSink** | Network | M | Multicast sender |
| [ ] **HttpSrc** | Network | M | HTTP GET source |
| [ ] **HttpSink** | Network | M | HTTP POST/PUT sink |
| [ ] **WebSocketSrc** | Network | M | WebSocket source |
| [ ] **WebSocketSink** | Network | M | WebSocket sink |

---

## Tier 4: High Value, Medium-High Complexity (Zenoh Integration)

Critical for distributed Zenoh-based pipelines.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **ZenohSrc** | Network | M | Subscribe to Zenoh topic |
| [ ] **ZenohSink** | Network | M | Publish to Zenoh topic |
| [ ] **ZenohQueryable** | Network | H | Zenoh queryable element |
| [ ] **ZenohQuerier** | Network | M | Zenoh query element |

---

## Tier 5: Medium Value, Low-Medium Complexity (Data Processing)

Common data manipulation operations.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [x] **StreamIdDemux** | Routing | M | Demux by stream ID |
| [ ] **FlatMap** | Transform | M | Map with multiple outputs |
| [ ] **DuplicateFilter** | Filter | M | Remove duplicates |
| [ ] **RangeFilter** | Filter | L | Filter by value range |
| [ ] **RegexFilter** | Filter | M | Filter by regex |
| [ ] **MetadataFilter** | Filter | L | Filter by metadata |
| [ ] **MetadataExtract** | Metadata | L | Extract to sideband |
| [ ] **BufferSplit** | Buffer | M | Split at delimiter |
| [ ] **BufferJoin** | Buffer | M | Join with delimiter |
| [ ] **BufferPad** | Buffer | L | Pad to fixed size |
| [ ] **BufferConcat** | Buffer | L | Concatenate contents |

---

## Tier 6: Medium Value, Medium Complexity (Compression)

Important for bandwidth-constrained scenarios.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **Lz4Encode** | Compression | M | LZ4 compression (fastest) |
| [ ] **Lz4Decode** | Compression | M | LZ4 decompression |
| [ ] **ZstdEncode** | Compression | M | Zstd compression (best ratio) |
| [ ] **ZstdDecode** | Compression | M | Zstd decompression |
| [ ] **GzipEncode** | Compression | M | Gzip compression |
| [ ] **GzipDecode** | Compression | M | Gzip decompression |
| [ ] **SnappyEncode** | Compression | M | Snappy compression |
| [ ] **SnappyDecode** | Compression | M | Snappy decompression |

---

## Tier 7: Medium Value, Medium Complexity (Serialization)

Format conversion for interoperability.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **JsonEncode** | Serialization | L | Serialize to JSON |
| [ ] **JsonDecode** | Serialization | M | Deserialize from JSON |
| [ ] **RkyvEncode** | Serialization | M | Serialize to rkyv |
| [ ] **RkyvDecode** | Serialization | M | Deserialize from rkyv |
| [ ] **CborEncode** | Serialization | M | Serialize to CBOR |
| [ ] **CborDecode** | Serialization | M | Deserialize from CBOR |
| [ ] **MsgPackEncode** | Serialization | M | Serialize to MessagePack |
| [ ] **MsgPackDecode** | Serialization | M | Deserialize from MessagePack |
| [ ] **ProtobufEncode** | Serialization | H | Serialize to protobuf |
| [ ] **ProtobufDecode** | Serialization | H | Deserialize from protobuf |

---

## Tier 8: Medium Value, Medium-High Complexity (Advanced Routing)

Sophisticated routing patterns.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **Router** | Routing | M | Route by content |
| [ ] **TopicRouter** | Routing | M | Route by topic/key |
| [ ] **LoadBalancer** | Routing | M | Round-robin, least-loaded |
| [ ] **Failover** | Routing | H | Health-checked failover |
| [ ] **Zip** | Routing | M | Pair from two streams |
| [ ] **ZipLatest** | Routing | M | Pair with latest |
| [ ] **Join** | Routing | H | Join by key |
| [ ] **TemporalJoin** | Routing | H | Join by timestamp |
| [ ] **Merge** | Routing | M | Merge sorted streams |

---

## Tier 9: Medium Value, High Complexity (Windowing & Aggregation)

Complex stateful processing.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **Window** | Transform | H | Tumbling/sliding/session windows |
| [ ] **Aggregate** | Transform | H | Aggregate over window |
| [ ] **Interleave** | Routing | H | Interleave by timestamp |
| [ ] **Deinterleave** | Routing | M | Split interleaved stream |
| [ ] **Muxer** | Routing | H | Combine with timing |
| [ ] **Demuxer** | Routing | H | Split combined stream |

---

## Tier 10: Medium Value, Medium Complexity (Debugging & Observability)

Essential for development and operations.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **LatencyProbe** | Debug | M | Measure latency |
| [ ] **ThroughputProbe** | Debug | M | Measure throughput |
| [ ] **BufferProbe** | Debug | L | Inspect/log buffers |
| [ ] **FakeSrc** | Debug | L | Generate fake data |
| [ ] **FakeSink** | Debug | L | Configurable consumer |
| [ ] **Recorder** | Debug | M | Record with timing |
| [ ] **Replayer** | Debug | M | Replay with timing |
| [ ] **Looper** | Debug | L | Loop N times |

---

## Tier 11: Lower Value, Medium Complexity (Security)

Important for secure pipelines but not always needed.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **HashCompute** | Security | L | Compute hash (SHA256, Blake3) |
| [ ] **HashVerify** | Security | M | Verify hash |
| [ ] **Checksum** | Security | L | Add checksum |
| [ ] **HmacSign** | Security | M | HMAC signing |
| [ ] **HmacVerify** | Security | M | HMAC verification |
| [ ] **AesEncrypt** | Security | M | AES encryption |
| [ ] **AesDecrypt** | Security | M | AES decryption |
| [ ] **ChaChaEncrypt** | Security | M | ChaCha20-Poly1305 |
| [ ] **ChaChaDecrypt** | Security | M | ChaCha20-Poly1305 |
| [ ] **JwtEncode** | Security | M | JWT encoding |
| [ ] **JwtDecode** | Security | M | JWT decoding |

---

## Tier 12: Lower Value, Various Complexity (Timing & Sync)

Specialized timing controls.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **Jitter** | Timing | L | Add random jitter (testing) |
| [ ] **Throttle** | Timing | M | Rate limiting with timing |
| [ ] **Pacer** | Timing | M | Pace to timestamps |
| [ ] **Sync** | Timing | H | Synchronize to clock |
| [ ] **LatencyTracer** | Timing | M | End-to-end latency |
| [ ] **ClockProvider** | Timing | H | Pipeline clock source |
| [ ] **TimestampSync** | Timing | H | Sync multiple streams |

---

## Tier 13: Lower Value, Medium Complexity (Chaos Testing)

Useful for resilience testing.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **DropProbe** | Debug | L | Randomly drop buffers |
| [ ] **CorruptProbe** | Debug | M | Randomly corrupt buffers |

---

## Tier 14: Niche, Various Complexity (Advanced Network)

Specialized network scenarios.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **TcpServerSrc** | Network | H | Multi-client server source |
| [ ] **TcpServerSink** | Network | H | Multi-client server sink |
| [ ] **TcpMux** | Network | H | Multiplex over TCP |
| [ ] **HttpServerSrc** | Network | H | HTTP server source |
| [ ] **HttpServerSink** | Network | H | HTTP server sink |
| [ ] **RtpSrc** | Network | H | RTP receiver |
| [ ] **RtpSink** | Network | H | RTP sender |
| [ ] **RtpJitterBuffer** | Network | H | RTP jitter buffer |

---

## Tier 15: Niche, Various Complexity (File Handling)

Specialized file operations.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **MultiFileSrc** | Source | M | Multiple files sequentially |
| [ ] **MultiFileSink** | Sink | M | File rotation/splitting |
| [ ] **SplitFileSrc** | Source | M | Split file segments |
| [ ] **SplitFileSink** | Sink | M | Split file segments |
| [ ] **Queue2** | Transform | H | File-backed queue |
| [ ] **TypeFind** | Transform | H | Auto-detect content type |
| [ ] **Capsfilter** | Transform | M | Filter by capabilities |

---

## Tier 16: Future (Media-Specific)

Reserved for potential media processing expansion.

| Element | Type | Complexity | Description |
|---------|------|------------|-------------|
| [ ] **AudioResample** | Audio | H | Resample audio |
| [ ] **AudioConvert** | Audio | H | Convert audio format |
| [ ] **AudioMixer** | Audio | H | Mix audio streams |
| [ ] **Volume** | Audio | M | Adjust volume |
| [ ] **AudioLevel** | Audio | M | Measure levels |
| [ ] **VideoScale** | Video | H | Scale video |
| [ ] **VideoConvert** | Video | H | Convert video format |
| [ ] **VideoRate** | Video | M | Adjust frame rate |
| [ ] **VideoOverlay** | Video | H | Overlay graphics |
| [ ] **VideoCompositor** | Video | H | Composite streams |

---

## Implementation Summary

### Completed: 25+ elements
- Core infrastructure (sources, sinks, transforms)
- Basic routing (Tee, Funnel, Selectors, Concat)
- Network (TCP, UDP - sync and async)
- Application integration (AppSrc, AppSink)

### Next Priority (Tier order):
1. **Quick wins** (Tier 1): Identity, MemorySrc/Sink, Delay, Metadata elements
2. **Data processing** (Tier 2): Filter, Map, Batch, Unbatch, Chunk
3. **Unix sockets** (Tier 3): UnixSrc, UnixSink
4. **Zenoh** (Tier 4): ZenohSrc, ZenohSink
5. **Compression** (Tier 6): Lz4, Zstd (highest value compression)

### Effort Estimates

| Complexity | Typical Effort | Examples |
|------------|----------------|----------|
| Low (L) | 1-2 hours | PassThrough, NullSink, Delay |
| Medium (M) | 2-4 hours | Queue, TcpSrc, Batch |
| High (H) | 4-8+ hours | Window, RtpJitterBuffer, Join |

---

## Notes

- Implement elements in tier order for maximum incremental value
- Each element should have comprehensive unit tests
- Consider both sync and async variants where applicable
- Backpressure handling is critical for all elements
- Document capabilities, limitations, and examples
