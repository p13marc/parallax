# Getting Started

This guide takes you from an empty project to a running pipeline with custom elements.

## Requirements

- **Linux.** Parallax uses `memfd_create`, `SCM_RIGHTS` fd passing, and `eventfd`; there is no macOS/Windows support.
- **Rust 1.85+** (edition 2024).

## Installation

```toml
[dependencies]
parallax-pipeline = "0.1"   # lib name is `parallax`: code writes `use parallax::...`
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The package is published as `parallax-pipeline` (the bare `parallax` name on
crates.io belongs to an unrelated crate), but the library target is named
`parallax`, so all imports use `parallax::`.

Default features are empty. Codecs, containers, network protocols, and device capture are all opt-in feature flags — see the [feature table in the README](../README.md#feature-flags).

## Your first pipeline

The quickest way to run a pipeline is the string syntax:

```rust
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> parallax::Result<()> {
    let mut pipeline = Pipeline::parse(
        "videotestsrc width=320 height=240 num-buffers=60 ! videoconvert ! nullsink",
    )?;
    pipeline.run().await
}
```

`Pipeline::parse` understands a linear chain of elements separated by `!`, each with optional `name=value` properties. See [pipeline.md](pipeline.md#parse-syntax) for the exact grammar, its limitations, and the list of built-in element names.

## Writing a custom source and sink

Elements implement one of the author traits from `parallax::element`. Sources write into a pre-allocated buffer provided through `ProduceContext`; sinks read through `ConsumeContext`:

```rust
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;
use parallax::Result;

struct Counter {
    current: u32,
    max: u32,
}

impl Source for Counter {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos); // end of stream
        }
        self.current += 1;
        let bytes = self.current.to_le_bytes();
        ctx.output()[..4].copy_from_slice(&bytes);
        ctx.set_sequence(self.current as u64);
        Ok(ProduceResult::Produced(4)) // wrote 4 bytes into the provided buffer
    }
}

struct Printer;

impl Sink for Printer {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&ctx.input()[..4]);
        println!("got {}", u32::from_le_bytes(bytes));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // A shared-memory arena backs the source's buffers: 8 slots of 64 bytes.
    let arena = SharedArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("counter", Counter { current: 0, max: 10 }, arena);
    let sink = pipeline.add_sink("printer", Printer);
    pipeline.link(src, sink)?;

    pipeline.run().await
}
```

Key points:

- `produce` returns a `ProduceResult`:
  - `Produced(n)` — wrote `n` bytes into the context's buffer (the zero-allocation fast path),
  - `OwnBuffer(buffer)` — the source made its own `Buffer` (fallback),
  - `OwnDmaBuf(buffer)` — a DMA-BUF backed buffer (zero-copy device path),
  - `WouldBlock` — no data right now,
  - `Eos` — stream finished.
- Transforms implement `Element` (`process(&mut self, Buffer) -> Result<Option<Buffer>>`) or the multi-output `Transform` trait. There are async variants (`AsyncSource`, `AsyncSink`, `AsyncTransform`) for real I/O.
- At end-of-stream the executor calls `flush()` repeatedly — implement it if your element buffers data (encoders, muxers, batchers).

## The simpler element API

If you don't need pooled buffers or contexts, the unified "simple" traits remove all boilerplate:

```rust
use parallax::element::{ProcessOutput, SimpleSink, SimpleSource, Snk, Src};

struct Hello { sent: bool }

impl SimpleSource for Hello {
    fn produce(&mut self) -> parallax::Result<ProcessOutput> {
        // construct and return a Buffer, or:
        Ok(ProcessOutput::Eos)
    }
}

struct Log;

impl SimpleSink for Log {
    fn consume(&mut self, buffer: &parallax::buffer::Buffer) -> parallax::Result<()> {
        println!("{} bytes", buffer.len());
        Ok(())
    }
}

let mut pipeline = parallax::pipeline::Pipeline::new();
let src = pipeline.add_element("src", Src(Hello { sent: false }));
let sink = pipeline.add_element("sink", Snk(Log));
pipeline.link(src, sink)?;
```

Wrap simple implementations in `Src(...)`, `Xfm(...)`, or `Snk(...)` and add them with `add_element`.

## Typed pipelines

For data processing where you want the compiler to check types between stages:

```rust
use parallax::typed::{pipeline, from_iter, map, filter, collect};

let result = (pipeline(from_iter(1..=10))
    >> filter(|x: &i32| x % 2 == 0)
    >> map(|x: i32| x * 10))
    .sink(collect::<i32>())
    .run()?
    .into_inner();
```

The `>>` operator is sugar for `.then(...)`. Available operators, sources, sinks, and multi-source combinators (`zip`, `merge`, `join`, `temporal_join`) are listed in [api.md](api.md#typed-pipelines).

## Using buffer pools

For steady-state streaming, pre-allocate buffers in a pool so the hot path never allocates:

```rust
use parallax::memory::FixedBufferPool;

let pool = FixedBufferPool::new(1024 * 1024, 10)?; // 10 × 1 MiB
let src = pipeline.add_source_with_pool("src", my_source, pool);
```

Inside `produce()`, call `ctx.acquire_buffer()` — it blocks when the pool is exhausted, giving you backpressure for free. See `examples/11_buffer_pool.rs`.

This is for **sources** only: blocking parks a thread, which a source can afford
and an element task cannot. Elements that allocate their own output buffers get
sized by the executor instead — see [memory.md](memory.md#output-arenas).

## Watching pipeline messages

```rust
let mut pipeline = Pipeline::parse("videotestsrc num-buffers=100 ! nullsink")?;
pipeline
    .run_with_bus(|msg| {
        println!("[{}] {}", msg.source, msg.kind);
        true // keep handling
    })
    .await?;
```

The bus carries `Eos`, `Error`, `Warning`, `Tag`, `Buffering`, `Qos`, and more — including a `futures::Stream` adapter for `select!` loops. Details in [pipeline.md](pipeline.md#bus--messages).

## Where to go next

- **Examples** — `examples/` contains 41 numbered, single-concept programs. Start with `01_hello` and work up; `cargo run --example 01_hello`.
- **[pipeline.md](pipeline.md)** — everything about constructing and controlling pipelines.
- **[memory.md](memory.md)** — the shared-memory model and cross-process pipelines.
- **[elements.md](elements.md)** — the full element catalog.
- **[scheduling.md](scheduling.md)** — hybrid real-time scheduling for low-latency audio/video.
