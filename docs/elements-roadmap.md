# Elements Roadmap

This document tracks the implementation status of pipeline elements, organized by category. Elements are inspired by GStreamer core elements and common streaming/media processing patterns.

## Legend

- [x] Implemented
- [ ] Not yet implemented

---

## Core Infrastructure

### Sources

- [x] **NullSource** - Produces empty buffers (testing)
- [x] **DataSrc** - Generates buffers from inline data
- [x] **TestSrc** - Test pattern generator (Zero, Ones, Counter, Random, Alternating, Sequence)
- [x] **AppSrc** - Inject buffers from application code
- [x] **FileSrc** - Read from file
- [x] **FdSrc** - Read from raw file descriptor
- [ ] **MemorySrc** - Read from memory buffer/slice
- [ ] **MultiFileSrc** - Read from multiple files sequentially
- [ ] **SplitFileSrc** - Read from split file segments

### Sinks

- [x] **NullSink** - Discards all buffers (benchmarking)
- [x] **AppSink** - Extract buffers to application code
- [x] **FileSink** - Write to file
- [x] **FdSink** - Write to raw file descriptor
- [x] **ConsoleSink** - Print buffer info to console (debugging)
- [ ] **MultiFileSink** - Write to multiple files (rotation, splitting)
- [ ] **SplitFileSink** - Write to split file segments
- [ ] **MemorySink** - Write to memory buffer

### Transforms

- [x] **PassThrough** - Identity element (no modification)
- [x] **Queue** - Async buffering with backpressure and leaky modes
- [x] **Valve** - On/off flow control switch
- [x] **RateLimiter** - Throughput limiting (buffers/sec, bytes/sec, fixed delay)
- [ ] **Queue2** - File-backed queue for large buffers
- [ ] **TypeFind** - Auto-detect buffer content type
- [ ] **Capsfilter** - Filter by capabilities/metadata

### Routing & Multiplexing

- [x] **Tee** - Duplicate to multiple outputs (1-to-N fanout)
- [x] **Funnel** - Merge multiple inputs (N-to-1 interleaved)
- [x] **InputSelector** - Select one of N inputs (N-to-1 switching)
- [x] **OutputSelector** - Route to one of N outputs (1-to-N routing)
- [x] **Concat** - Sequential stream concatenation
- [x] **StreamIdDemux** - Demultiplex by stream ID
- [ ] **Muxer** - Combine multiple streams with timing
- [ ] **Demuxer** - Split combined stream
- [ ] **Interleave** - Interleave multiple streams by timestamp
- [ ] **Deinterleave** - Split interleaved stream

---

## Network Elements

### TCP

- [x] **TcpSrc** - Read from TCP connection (client/server modes)
- [x] **TcpSink** - Write to TCP connection
- [x] **AsyncTcpSrc** - Async TCP source
- [x] **AsyncTcpSink** - Async TCP sink
- [ ] **TcpServerSrc** - Multi-client TCP server source
- [ ] **TcpServerSink** - Multi-client TCP server sink
- [ ] **TcpMux** - Multiplex multiple streams over TCP

### UDP

- [x] **UdpSrc** - Receive UDP datagrams
- [x] **UdpSink** - Send UDP datagrams
- [x] **AsyncUdpSrc** - Async UDP source
- [x] **AsyncUdpSink** - Async UDP sink
- [ ] **UdpMulticastSrc** - Multicast UDP receiver
- [ ] **UdpMulticastSink** - Multicast UDP sender
- [ ] **RtpSrc** - RTP packet receiver
- [ ] **RtpSink** - RTP packet sender
- [ ] **RtpJitterBuffer** - RTP jitter buffer for reordering

### Unix Sockets

- [ ] **UnixSrc** - Read from Unix domain socket
- [ ] **UnixSink** - Write to Unix domain socket

### HTTP/WebSocket

- [ ] **HttpSrc** - HTTP client source (GET)
- [ ] **HttpSink** - HTTP client sink (POST/PUT)
- [ ] **HttpServerSrc** - HTTP server source (receive uploads)
- [ ] **HttpServerSink** - HTTP server sink (serve content)
- [ ] **WebSocketSrc** - WebSocket client/server source
- [ ] **WebSocketSink** - WebSocket client/server sink

### Zenoh Integration

- [ ] **ZenohSrc** - Subscribe to Zenoh topic
- [ ] **ZenohSink** - Publish to Zenoh topic
- [ ] **ZenohQueryable** - Zenoh queryable element
- [ ] **ZenohQuerier** - Zenoh query element

---

## Data Processing Elements

### Filtering

- [ ] **Filter** - Generic predicate-based filter
- [ ] **RangeFilter** - Filter by value range
- [ ] **RegexFilter** - Filter by regex pattern match
- [ ] **SampleFilter** - Statistical sampling (every Nth, random %)
- [ ] **DuplicateFilter** - Remove duplicate buffers
- [ ] **ThrottleFilter** - Rate-based filtering

### Transformation

- [ ] **Map** - Apply function to buffer contents
- [ ] **FlatMap** - Map with multiple output buffers
- [ ] **Batch** - Combine N buffers into one
- [ ] **Unbatch** - Split one buffer into N
- [ ] **Chunk** - Split into fixed-size chunks
- [ ] **Aggregate** - Aggregate buffers over window
- [ ] **Window** - Windowing (tumbling, sliding, session)

### Buffer Manipulation

- [ ] **BufferSplit** - Split buffer at delimiter
- [ ] **BufferJoin** - Join buffers with delimiter
- [ ] **BufferPad** - Pad buffer to fixed size
- [ ] **BufferTrim** - Trim buffer to max size
- [ ] **BufferSlice** - Extract slice from buffer
- [ ] **BufferConcat** - Concatenate buffer contents

### Metadata

- [ ] **MetadataInject** - Add/modify buffer metadata
- [ ] **MetadataExtract** - Extract metadata to sideband
- [ ] **MetadataFilter** - Filter by metadata values
- [ ] **Timestamper** - Add/modify timestamps
- [ ] **SequenceNumber** - Add sequence numbers

---

## Timing & Synchronization

### Timing

- [ ] **Delay** - Fixed delay element
- [ ] **Jitter** - Add random jitter (testing)
- [ ] **Timeout** - Timeout with fallback
- [ ] **Debounce** - Debounce rapid buffers
- [ ] **Throttle** - Rate limiting with timing
- [ ] **Pacer** - Pace output to timestamps

### Synchronization

- [ ] **Sync** - Synchronize to clock
- [ ] **LatencyTracer** - Measure end-to-end latency
- [ ] **ClockProvider** - Pipeline clock source
- [ ] **TimestampSync** - Synchronize multiple streams

---

## Compression & Encoding

### Compression

- [ ] **GzipEncode** - Gzip compression
- [ ] **GzipDecode** - Gzip decompression
- [ ] **ZstdEncode** - Zstandard compression
- [ ] **ZstdDecode** - Zstandard decompression
- [ ] **Lz4Encode** - LZ4 compression
- [ ] **Lz4Decode** - LZ4 decompression
- [ ] **SnappyEncode** - Snappy compression
- [ ] **SnappyDecode** - Snappy decompression

### Serialization

- [ ] **JsonEncode** - Serialize to JSON
- [ ] **JsonDecode** - Deserialize from JSON
- [ ] **RkyvEncode** - Serialize to rkyv
- [ ] **RkyvDecode** - Deserialize from rkyv
- [ ] **ProtobufEncode** - Serialize to protobuf
- [ ] **ProtobufDecode** - Deserialize from protobuf
- [ ] **CborEncode** - Serialize to CBOR
- [ ] **CborDecode** - Deserialize from CBOR
- [ ] **MsgPackEncode** - Serialize to MessagePack
- [ ] **MsgPackDecode** - Deserialize from MessagePack

---

## Cryptography & Security

### Hashing

- [ ] **HashCompute** - Compute hash (SHA256, Blake3, etc.)
- [ ] **HashVerify** - Verify hash
- [ ] **Checksum** - Add checksum to buffer

### Encryption

- [ ] **AesEncrypt** - AES encryption
- [ ] **AesDecrypt** - AES decryption
- [ ] **ChaChaEncrypt** - ChaCha20-Poly1305 encryption
- [ ] **ChaChaDecrypt** - ChaCha20-Poly1305 decryption

### Authentication

- [ ] **HmacSign** - HMAC signing
- [ ] **HmacVerify** - HMAC verification
- [ ] **JwtEncode** - JWT encoding
- [ ] **JwtDecode** - JWT decoding

---

## Debugging & Testing

### Debugging

- [ ] **Identity** - Pass-through with callbacks (debugging)
- [ ] **FakeSrc** - Generate fake data at specified rate
- [ ] **FakeSink** - Consume with configurable behavior
- [ ] **LatencyProbe** - Measure latency at point
- [ ] **ThroughputProbe** - Measure throughput at point
- [ ] **BufferProbe** - Inspect buffers (logging/metrics)
- [ ] **DropProbe** - Randomly drop buffers (chaos testing)
- [ ] **CorruptProbe** - Randomly corrupt buffers (chaos testing)

### Recording & Replay

- [ ] **Recorder** - Record stream to file with timing
- [ ] **Replayer** - Replay recorded stream with timing
- [ ] **Looper** - Loop stream N times or infinitely

---

## Advanced Routing

### Content-Based Routing

- [ ] **Router** - Route based on buffer content
- [ ] **TopicRouter** - Route based on topic/key
- [ ] **LoadBalancer** - Load-balanced routing (round-robin, least-loaded)
- [ ] **Failover** - Failover routing with health checks

### Stream Joining

- [ ] **Zip** - Pair buffers from two streams (1:1)
- [ ] **ZipLatest** - Pair with latest from each stream
- [ ] **Join** - Join streams by key
- [ ] **TemporalJoin** - Join by timestamp proximity
- [ ] **Merge** - Merge sorted streams

---

## Media-Specific (Future)

### Audio

- [ ] **AudioResample** - Resample audio
- [ ] **AudioConvert** - Convert audio format
- [ ] **AudioMixer** - Mix multiple audio streams
- [ ] **Volume** - Adjust audio volume
- [ ] **AudioLevel** - Measure audio levels

### Video

- [ ] **VideoScale** - Scale video frames
- [ ] **VideoConvert** - Convert video format
- [ ] **VideoRate** - Adjust frame rate
- [ ] **VideoOverlay** - Overlay graphics on video
- [ ] **VideoCompositor** - Composite multiple video streams

---

## Implementation Priority

### Phase 1: Core (DONE)
Basic infrastructure for pipeline construction.

### Phase 2: Elements Foundation (DONE)
Core element traits and basic elements.

### Phase 3: GStreamer Equivalents (DONE)
Essential elements matching GStreamer core.

### Phase 4: Network & IPC (Current)
1. Unix sockets (UnixSrc, UnixSink)
2. HTTP elements (HttpSrc, HttpSink)
3. WebSocket elements
4. Zenoh integration

### Phase 5: Data Processing
1. Filtering elements (Filter, SampleFilter, DuplicateFilter)
2. Transformation (Map, Batch, Unbatch, Window)
3. Buffer manipulation

### Phase 6: Timing & Sync
1. Delay, Timeout, Debounce
2. Synchronization elements
3. Clock integration

### Phase 7: Encoding & Compression
1. Compression (Gzip, Zstd, Lz4)
2. Serialization (JSON, rkyv, Protobuf)

### Phase 8: Security
1. Hashing elements
2. Encryption/decryption
3. Authentication

### Phase 9: Debugging & Testing
1. Probes and identity elements
2. Recording/replay
3. Chaos testing elements

### Phase 10: Advanced Routing
1. Content-based routing
2. Load balancing
3. Stream joining (Zip, Join, Merge)

---

## Notes

- Each element should have comprehensive unit tests
- Elements should support both sync and async where appropriate
- Consider backpressure and flow control in all elements
- Document capabilities and limitations clearly
- Provide examples for complex elements
