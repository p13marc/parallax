# Changelog

Notable changes to `parallax-pipeline`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semver](https://semver.org/), with the pre-1.0 caveat that breaking changes
ride minor bumps.

Earlier releases predate this file — their notes live in the `Bump to X.Y.Z`
commit messages and the Forgejo releases.

## [Unreleased]

Every out-of-tree plugin must be rebuilt: the plugin ABI went **12 → 14**.

### Added

- **`parallax::video` — presentation maths without a graphics dependency**
  (#263). Plane strides, the letterbox rect, the YUV matrices, the WGSL and
  the `(format, matrix) → shader mode` table now live in a public, unfeatured
  module, over a graphics-free `TexelFormat` instead of `wgpu::TextureFormat`.
  That is what lets parallax's own `display-gpu` backend and a GUI host on a
  *different wgpu major* share the arithmetic and the golden tests that pin it.
  `FrameView::from_metadata` is the one read of a buffer's format, geometry and
  strides — `AutoVideoSink` uses it, so a host presenter cannot disagree with
  the sink about what a buffer is. Thirteen of the twenty-two tests here
  previously ran only under `--features display`; they run in the default
  `cargo test` job now. No golden value changed.

- **`AppVideoSink` — a paced sink that hands frames to a GUI** (#264).
  `AutoVideoSink` builds the process's one winit event loop, so an application
  that owns that loop itself cannot use it. This one keeps everything needing a
  `ConsumeContext` — the anchor handshake, segment mapping, flush generations,
  out-of-segment clipping, QoS — and publishes each frame into a `FrameMailbox`
  stamped with the running time it is due at; the host decides when to show it,
  because only the host knows when the next vertical blank is. Unfeatured: no
  winit, no wgpu. Back-pressure is the wait for mailbox space, which is what
  makes the pipeline run at playback speed — **bounded**, because an occluded
  window stops redrawing and blocking there would starve the audio branch and
  freeze the master clock. See `examples/64_appvideosink.rs`.

- **`PipelineHandle::dispatch_seek` / `dispatch_event_upstream`** (#272), the
  synchronous twins of `seek` / `send_event_upstream`. Dispatching an upstream
  event never blocked — every channel it touches is unbounded, and what takes
  time is the flush trio running elsewhere — but a GUI event handler is neither
  async nor able to move a `PipelineHandle` into a task, so the async versions
  now call these rather than leaving callers to rely on the fact.

- **Two new workspace crates**, both `publish = false`: `parallax-iced` (a
  `shader` widget that draws a `FrameMailbox`, on Iced's wgpu) and the
  rewritten `parallax-player` built on it. The crossterm/winit demo that held
  that name is gone (#277) — the new one took the name and the binary path, so
  `scripts/decode_matrix.py` and every `cargo run -p parallax-player` are
  unchanged. `just check-iced`, `just play-headless` (the real widget under
  weston) and `just play-smoke` (`--headless`, no compositor) are its gates.

- **`parallax-player --headless`** (#277) plays with no window at all: this
  process becomes the host and drains the frame mailbox at the same due times,
  with the same lateness budget, that the widget uses. It needs no compositor,
  no GPU and no sound card, so CI runs the player end to end for the first
  time. It reports the same frame counts the windowed run does — that
  agreement is the point, and the reason it calls `take_due` and
  `parallax_iced::DEFAULT_MAX_LATENESS` rather than inventing a second pacing
  policy.

- **Subtitles** (#279). `parallax::subtitle` is a new unfeatured module beside
  `parallax::video`: a `Cue` (text plus its interval), a `CueTrack` that
  retains cues until they expire, `srt` for sidecar files and `text` for
  taking the words out of an ASS event body. `AppSubtitleSink` publishes into
  a `CueTrack` the way `AppVideoSink` publishes into a `FrameMailbox`, sharing
  its flush discipline; the player draws them with Iced's own text stack,
  bottom-centred, mounted only while a cue is on screen.

  **A cue is an interval, and that is why it needed its own store.**
  `FrameMailbox::take_due` pops and forgets — right for a frame, wrong for a
  subtitle, which would then show for exactly one repaint. `CueTrack::active`
  answers "what is on screen now" as many times as it is asked, and
  `retire()` clears the cue *being displayed* as well as the queue: a seek
  that left the old subtitle burnt over the new picture is precisely the bug
  the video path's generations exist to prevent.

  **Matroska never stores SRT as SRT.** A muxer converts it to `S_TEXT/UTF8`
  with the timing in the container, so a subtitle track inside an MKV needs no
  parsing at all — the block payload *is* the cue text and `BlockDuration`
  *is* the out-time. The SRT parser exists for the sidecar `.srt` next to the
  video, which is where a downloaded subtitle actually lives. ASS is degraded
  to plain text (eight commas to the event body, then `{...}` overrides
  stripped and `\N` turned into a line break) and says so; PGS and VobSub are
  bitmap formats and are refused rather than shown as garbage.

  Subtitles are **off until asked for**: a container's default-subtitle flag
  is a hint about a language preference the engine does not have.

- **`MediaFormat::Text(TextCodec)`** (#279), with its wire code and its arm in
  both projections. A cue that crossed a process or machine boundary as
  anonymous bytes is the drift #219 and #238 exist to prevent. `WIRE_VERSION`
  is 6 and the **plugin ABI is 14**: caps cross the plugin boundary by value,
  so a plugin compiled against the old enum could be handed a discriminant it
  has no arm for — the same reason `MemoryType::External` minted 11.

- **Track enumeration and selection** (#278). `Mp4Track` and `MkvTrack` carry
  a `TrackLabel` — language, name, and the default/forced/enabled flags — so a
  picker has something to label a row with. MKV had all of it available and
  unread; MP4 needed three sources (`mdhd`'s packed language code, `hdlr`'s
  name where several muxers write a title, and `tkhd`'s enabled bit), and its
  missing pieces are an honest `None` rather than something invented: there is
  no forced flag in MP4 at all, and `elng` is not modelled by the crate.

  `MkvDemux::with_tracks` and `Mp4DemuxSource::with_tracks` select by
  container id, falling back to the container's own default flag and then to
  the first playable track. A track id that does not exist, or names a track
  of the wrong type, is an **error** — a picker offered it, so a silent
  fall-back would hide a real bug. `MkvDemux::route` is public with them.

  Selection is construction-time by design, and the player switches tracks by
  rebuilding and resuming where it was: an AAC track and an E-AC-3 track need
  different decoder elements, so a switch is a different graph rather than a
  re-pointed pad. `--video-track` / `--audio-track` and a picker in the
  diagnostics panel.

- **`MessageKind::StreamCollection` and `MessageKind::Tag` have producers**
  (#278). Both, plus `StreamInfo`, `BusHandle::post_tags` and all of
  `pipeline::tags`, shipped as types with no caller anywhere in the tree. The
  file demuxers now post one collection and one tag list on their first
  `produce` — a track list with each stream's language, flags and whether it
  is the one playing, and the container format, duration and selected codecs.
  `Demuxer` gained a defaulted `set_bus` to reach the bus at all; the executor
  already called `set_bus` on every element, and the adapter simply had
  nowhere to forward it to. No ABI change.

- **`TagValue::Bool`** and the `track-default` / `track-forced` /
  `track-selected` tag keys (#278). `StreamInfo` has no field for which stream
  is playing, and a collection that cannot say that is half an answer.

- **Audio output device selection** (#284). `--audio-device NAME` picks it,
  `--audio-device list` prints what ALSA offers, and the diagnostics panel
  has a picker; choosing one rebuilds the pipeline and resumes where it was,
  because the graph is fixed at `start()` and an ALSA sink is opened once for
  one device.

- **`PipeWireSink::audio(Some(name))` reaches the named node** (#284). The
  device parameter was carried as far as `playback_thread` and then dropped —
  the parameter was literally named `_device`, and the stream connected with
  `AUTOCONNECT` and no target, so a named device played out of whatever the
  session manager chose. It sets `TARGET_OBJECT` now, and so does the capture
  side, which had the same hole one indirection later.

- **Snapshot the frame on screen** (#283). `c` writes the *decoder's* frame —
  full resolution, source colour, no scaling, letterbox or overlay — through
  the same `VideoConvert` the pipeline uses, so a saved still and the picture
  it was taken from agree about the matrix and range the stream declared.

  Not the pipeline branch the issue proposes, and `snapshot.rs` says why:
  a closed `Valve` *drops* rather than queues and has no one-shot mode, so
  opening and closing it from the interface thread lets an unbounded number
  of frames through; `FileSink` cannot be re-targeted and appends, so N
  snapshots would concatenate into one broken PNG; and `Valve::passthrough()`
  is true, which means the branch adds its whole downstream capacity to the
  decoder's hold rather than costing nothing while idle. The host already has
  the frame.

- **`FrameMailbox::last_presented()`** (#283) — the frame the host last took,
  kept for whoever wants it afterwards. A `PacedFrame` clone moves no pixels,
  and the slot it holds is one a host displaying a frame retains anyway. It
  survives a flush on purpose: the frame on screen after a seek is still the
  frame on screen.

- **Frame stepping** (#281). `.` and `,` step one frame forward or back,
  pausing first. It is a flushing `ACCURATE` seek per frame — wasteful, and
  correct: `plans/30_iced_player.md` carries the design for the real
  `Event::Step` that a paced sink would consume without a flush.

  Two details decide whether it works at all. `SeekEvent::new_time` is
  `FLUSH | KEY_UNIT` and `with_flags` *replaces* rather than ORs, so a step
  restates the whole set — and it must add **`SNAP_BEFORE`** as well as
  `ACCURATE`, because MKV's default snap is *forward*: without it a step from
  600 ms to 700 ms lands on the keyframe at 1000 ms, skipping most of a GOP in
  the name of advancing one frame. And the step size comes from the container
  header, carried on the player's `VideoStream`, because nothing on the buffer
  path has a framerate — the demuxers emit encoded buffers, whose
  `MediaFormat::Video` has no slot for one, and `Framerate::default()` is
  30 fps, so a decoded frame cannot be asked either. A file that states no
  frame rate refuses to step rather than guessing 33 ms and calling it a frame.

- **The gestures a GUI is expected to have** (#274). Wheel to change the
  volume, a pointer that disappears with the chrome (under the same rule, so
  the film is alone on screen when it is playing), `o` for a file dialog
  through the XDG portal, files dropped on the window, and a list of what was
  played lately on the empty screen. `Esc` leaves fullscreen rather than
  entering it; `f` is still the toggle.

  The dialog is `ashpd`, not `rfd`: parallax already depends on ashpd 0.13 for
  the screencast portal, and `rfd` would pull a second major of it into the
  same binary — `just check-iced` now refuses that the way it refuses two
  wgpus. Drag-and-drop is deliberately the garnish rather than the mechanism:
  Iced's `window::Event::FileDropped` says "**Wayland:** Not implemented" in
  its own documentation, and this player enables both backends.

- **A playlist, and open-next** (#276). Several files on the command line, or
  dropped on the window, are a queue: it plays through them and stops, or
  repeats one or all. A file that will not open is marked and skipped rather
  than stopping the list, and a full lap with nothing playable stops rather
  than spinning.

  Advancing keys off the bus's `MessageKind::Eos`, which is an *edge* —
  posted once, after every task finished, and scoped to the engine whose bus
  carries it. `FrameMailbox::end_reason()` is a level that stays set, and the
  sink raises it when the EOS *event* arrives, with frames still queued and
  the audio device buffer still to play; `PipelineHandle::ended()` would
  outlive the engine it belongs to and fire again on every manual Next, since
  `stop()` reports `Eos` too.

  The open chain became a state machine because it had a race. `open()` took
  the engine and spawned a teardown, so a second Next 50 ms later found
  `engine == None`, took the "nothing to tear down" path, and built a second
  pipeline — two demuxers, two decoders and two ALSA devices — while the
  first was still winding down. And a probe answering a superseded request
  would install itself over a live engine, which leaks it outright:
  `PipelineHandle` has no `Drop`, so dropping one detaches its tasks. Now an
  engine exists only in `Stage::Playing`, exactly one teardown is ever in
  flight (N rapid presses collapse into one teardown and one probe of the
  last target), and a probe is built only if its seqnum is still the one being
  waited for.

- **The diagnostics panel says what the pipeline is doing** (#275). Which
  decoder actually won (`--hwdec auto` leaves that open, and the answer used
  to be logged once and forgotten), which adapter and backend Iced is drawing
  on, the elected clock, buffering and download progress, the newest QoS
  advisory, worst observed lateness, and a `d` key that writes the running
  graph out as GraphViz. Bus warnings and errors are **toasts** now rather
  than one sticky banner: a warning fades after eight seconds, an error waits
  to be dismissed, because a message that vanishes before it is read is the
  same as no message. `MessageKind::DurationChanged` finally reaches the seek
  bar — a file whose length only becomes known after the probe used to keep
  the "live / unknown length" label for the whole run.

- **A `feature-union` CI job** (#277). The retired player's Cargo.toml was the
  only build of `mp4-demux,mkv-demux,vpx,av1-decode,opus,h264,display,
  display-gpu,simd-colorspace,audio-aac,audio-vorbis,eac3,alsa,vaapi` in one
  dependency graph. Every feature in it is built by some other job; the
  *union* was not. Against the root crate with `--all-targets` it also
  compiles `present/dmabuf_import.rs`'s `hardware_tests` for the first time,
  which the old job could never reach — it built the root crate as a
  dependency, with `cfg(test)` off.

### Fixed

- **`ZenohSink` no longer turns on `express` for the `@media` frame profile**
  (#262). Both halves of the reasoning it was added under are true and neither
  of them is express: "a stale frame is worthless" is `Drop` plus
  `InteractiveHigh`, "the encoder must never block" is best-effort plus drop,
  and the profile already set all four. Express is a third, independent axis —
  *do not wait to batch* — and batching engages only under back-pressure, so it
  is a no-op on an unsaturated link and spends per-message framing at exactly
  the moment a `drop` profile should be shedding. zenkey RFC v1.26 M1 removes
  it from `frame` for that reason, keeping it on `alert`, and it was enough to
  stop a downstream consumer adopting the sink at all: it validates its
  `@media` plane against the convention, and `zenctl doctor` compares exactly
  these four axes. `with_express(true)` still overrides it, from the
  pipeline-string API too. `ZenohSink` gained `express()`,
  `congestion_control()` and `priority()`, because the profile was previously
  unobservable outside a packet capture.

- **A third demuxer pad would have been called "audio"** (#279). Both file
  demuxers' `pad_name` ended in `_ => "audio"`, so routing anything to
  `PadId(2)` would have sent it silently down the audio branch. Exhaustive
  now, in both.

- **MKV could not describe two video tracks** (#278). `nal_config` was a
  single `Option<NalConfig>` where MP4 has a per-track map, so routing a
  second video track overwrote the first one's parameter sets outright. It
  never bit because nothing could select a track; `with_tracks` would have
  made it reachable. Fixed with it rather than left as a landmine.

- **ALSA no longer plays at a rate the hardware is not using** (#284).
  `AlsaSink::new` asks for a rate with `ValueOr::Nearest`, so a driver that
  cannot do 44.1 kHz **substitutes** the closest it can and says nothing —
  while `self.format.sample_rate` kept the *requested* value. Everything
  derived from it then ran on a fiction: the pinned caps, the declared
  latency, `frames_to_nanos`, and `AlsaPosition`, which is the master clock
  the whole pipeline paces against. The result was audio at the wrong pitch
  and a clock that drifts, with no error anywhere.

  The granted rate, channels, buffer and period are read back after
  `hw_params` now. A rounded buffer is accepted and the real numbers stored,
  because that is normal and harmless once the latency follows it; a
  substituted *rate* is refused with `DeviceError::FormatNotSupported` — which
  was defined and never constructed in this file — naming both rates and
  saying to resample upstream. `AlsaSrc` had the same hole and gets the same
  treatment: its timestamps are derived from that rate too.

- **Playback speed survives a track change** (#276). `reset_transport` cleared
  position, pause and the seek state; `build` re-applied the volume. Nothing
  re-applied the rate, so opening a second file left the transport showing
  `2x` and the label saying "muted" while the new pipeline played at 1x.

- **A packed frame's length is measured by its own packing** (#264).
  The geometry read checked declared dimensions against a hardcoded four bytes
  per pixel. `AutoVideoSink` could assume that because its caps admit nothing
  else, but shared it made a 320×240 `Rgb24` frame fail the check, fall through
  to the size guess, and report 278×207.

### Breaking

- **A zenoh publish error drops the sample instead of ending the run** (#258).
  A failed `put` returned `Err` from `ZenohSink::consume`, which the sink task
  turned into `EndReason::Error` — so one transport fault ended a 24/7
  publisher, and `publish_errors` was structurally bounded by one, which is why
  #246 gave it no rate. It now counts the failure, warns on a 1st/10th/100th
  ladder with a matching bus `Warning`, and carries on, for the same reason the
  executor sheds a `PoolExhausted`: a dropped frame beats a dead live pipeline,
  and on `@media` the frame was best-effort and droppable by contract anyway.
  `ZenohSink::with_fatal_publish_errors(true)` (`fatal-publish-errors=true`
  from a pipeline string) restores the old behaviour, which is what batch work
  wants. Failing to *declare* the publisher stays fatal either way — that is a
  configuration fault, not a hiccup. `publish_errors` is a real unbounded
  counter now, with `publish_errors_per_second()` beside it.

- **`ZenohSinkHandle::samples_sent`/`bytes_sent` are `samples_offered`/
  `bytes_offered`** (#289), with deprecated shims. The old names promised
  delivery the sink cannot observe: under `CongestionControl::Drop` — which the
  `@media` profile pins, correctly — zenoh discards a sample inside its
  transport when the queue is full and `put()` still returns `Ok`. Measured on
  a shaped veth lab at 1.69 Mbps offered, a receiver got 44 of 255 sequences
  over a 300 kbit link and 14 of 193 over a 100 kbit one, while every
  sender-side counter read zero. There is no fix inside the sink — zenoh counts
  transport drops only under its `stats` feature and only per *link*, never per
  publisher, and `Block` would trade the loss for a stall behind the slowest
  viewer — so delivery is a receiver-side measurement, and
  `ZenohSrcHandle::last_frame_age()` is the second axis it needed.
  `lost_wire()` says how much went missing; age says *why*, because the two
  modes differ by orders of magnitude: congestion backs samples up behind a
  full queue (medians of 3.5 s and 9.1 s in the same lab) while genuine
  in-flight loss discards frames without delaying the survivors (0.77 ms while
  losing 20 %). Only as good as the clock sync between the two machines, and
  said so.

- **A/V synchronization has one origin instead of two guesses** (#265).
  Running time used to be `clock.now() - base_time`, where `base_time` was a
  `ClockTime` snapshotted at `Executor::start` and copied into every element.
  A flushing seek restarts every segment at running time zero while an
  audio-provided master clock keeps counting monotonically upward, so after a
  seek the two quantities no longer named the same instant — and each sink
  papered over that on its own. `AutoVideoSink` pinned its first post-seek
  frame's mapped time to whatever running time it happened to *arrive* at;
  `AlsaSink` was entirely PTS-blind, with no `Event::Segment` arm at all, and
  simply wrote whatever bytes it was handed. The post-seek A/V offset was
  therefore the arrival skew between the two branches — a rebuilt openh264 and
  a keyframe decode on one side against a freshly emptied device accepting a
  buffer instantly on the other — and it stayed until the next seek re-rolled
  it. Measured on the test rig: −130 ms at start-up, −330 ms after one seek,
  430 ms worst across four.

  `base_time` is now a shared, re-anchorable `SyncBase`, reachable through
  `ProduceContext`/`ConsumeContext`. On the first in-segment buffer of each
  flush generation exactly one designated **sync master** — the sink whose
  element provided the selected clock — publishes
  `base = clock_when_that_sample_is_heard − segment.to_running_time(pts)`;
  every other sink reads it. A pipeline with no clock provider has no master
  and the first sink to see data anchors it, which is what the old private
  anchor did, only shared. A master that never arrives is waited out, so a
  stalled audio branch cannot freeze the picture. `AutoVideoSink` presents at
  `running_time >= segment.to_running_time(pts)` and nothing else.

  API breaks: `set_clock` (on `AsyncElementDyn`, the author traits and both
  contexts) takes a `SyncHandle` instead of a `ClockTime`; `base_time()` keeps
  its name but reads the live shared cell; `FlushStopEvent` grew the seek
  epoch that names the generation; `Demuxer` grew `take_downstream_event`.
- **`Mp4SeekPoint.time_ns` is a composition time**, not a decode time (#265).
  The seek tables are `stts`/`stss`, which are decode time, while `read_sample`
  stamps `pts` from composition time — so the landing that becomes the
  synthesized segment's `start`, the `SeekDone` position and the target every
  non-reference track is re-sought to was on the wrong clock. On any B-frame
  stream that put audio ahead of video by the reorder delay, 40–125 ms.
- **`MkvDemux` reports a forward seek's landing** instead of `None`, and uses
  it as the skip floor for every track (#265). Cues are video keyframes, so
  the landing is knowable up front; it used to start audio at the requested
  time and video at the next keyframe, up to a whole GOP apart.
- **Audio sinks drop out-of-segment audio.** `AlsaSink` and `PipeWireSink`
  clip head and tail sample-accurately; they previously played every byte they
  were handed. A container seeks its non-reference tracks at-or-before the
  reference keyframe on purpose, so the first post-seek audio buffer routinely
  straddles `segment.start`.
- **`with_loop` is no longer a silent rewind.** Both source-style demuxers
  stage a `Segment` at the lap boundary, so running time stays monotonic
  across it. Absolute pacing requires it: without the segment, lap two reads
  as uniformly late and the picture freezes for good.
- `AudioDecoder` gained a defaulted `reset()`. Out-of-tree decoders keep
  compiling, but any with inter-packet state should override it — it is what a
  flushing seek now calls.

### Fixed

- `AudioDecoderElement` had no `handle_downstream_event` at all, so a seek
  never reset the wrapped decoder, its running `current_pts` carried the
  abandoned position into the new one, and the executor's discarded
  `FlushStart` drain consumed the one-shot `flushed` latch — losing the real
  EOS tail from the first seek of a run onwards. A buffer legitimately stamped
  at `pts == 0` was also treated as unstamped.
- `AutoVideoSink` had no `FlushStart` arm, so up to four pre-seek frames
  queued for the display thread were blitted *after* the seek — the old
  picture flashing back over the new one.
- `PipeWireSink` had no event handling at all: a flushing seek left up to
  sixteen buffers of pre-seek audio to play out.
- A demuxer's pads each anchored their own segment, which forced every stream
  to running time zero at its own start and erased the offset *between* them.
  One origin per node now, shared by every pad.
- `TsDemuxElement`'s ACCURATE refinement judged whichever stream's PTS arrived
  first — normally audio, running ahead of video — rather than the video the
  segment and the picture are driven by.
- `AlsaSink` declared no latency, so a pipeline's audio path was reported as
  instantaneous; its ~85 ms device buffer is now in
  `PipelineHandle::latency()`. Its clock also no longer interpolates through a
  flush gap the device is not playing through, and a seek after EOS no longer
  leaves the master clock free-running at wall rate forever.
- `broadcast_eos`/`broadcast_event` fan out concurrently, as buffers already
  did: a branch that cannot accept the message no longer holds up a sibling
  that could — which deadlocked teardown in an A/V graph.
- `PipelineHandle::abort` recorded `Aborted` after raising the cooperative
  stop, leaving a window for a source to report a clean EOS for a torn-down
  run.
- **A flushing seek could deadlock a paced pipeline.** A sink inside its
  pacing wait cannot reach its own loop head, so it never sees the
  `FlushStart` queued behind the buffer it is holding — a buffer from the
  position the seek just abandoned. It waited out a superseded presentation
  time while its link filled, which backed the demuxer up, which starved the
  branch whose device drives the clock it was waiting on. `--loop` plus
  repeated seeks reached it in about half a second. `SyncBase::flush_epoch`
  now rises the instant a flushing seek is dispatched, and a wait that began
  before it drops its buffer — the pacing wait and the wait for the sync
  master alike. `StallWatch` presents after `STALL_LIMIT` of a motionless
  clock as a backstop, excluding a deliberate pause.
- `parallax-player` applied arrow presses to `handle.position()`, which still
  reports the old position until the seek lands, so a held key walked a target
  that kept being recomputed from where playback had not moved from yet and
  each press fired its own flush. Presses coalesce now, one seek is in flight
  at a time and is matched back by seqnum, and the status line shows the
  target it is heading for.

## [0.8.0] — 2026-08-25

63 commits since 0.7.0, and the release a parallax pipeline becomes a
*multi-machine* one: it publishes and consumes the ratified zenkey `@media`
plane, ingests RTSP well enough to leave running, and reports enough about
itself for a consumer to measure what it is getting.

Every out-of-tree plugin must be rebuilt: the plugin ABI went **11 → 12**.
Peers must be upgraded together: the zenoh attachment went **v1 → v4** and the
IPC descriptor **128 → 192 bytes**. Several wire-format breaks land in one
release on purpose, so that no intermediate build advertises a half-migrated
format.

### Breaking

- **`RtspSrc` no longer attaches `RtpMeta`** (#208). `Metadata.rtp` is now
  `None` on buffers from an RTSP source, where it used to be `Some` with three
  of its five fields zeroed. retina hands back demuxed access units, not
  packets, so `seq`, `ssrc` and `pt` were never available — and zeros are not
  "absent", they are plausible-looking garbage that anything reading them
  believed, including the IPC descriptor, which faithfully carries `RtpMeta`
  across a process boundary. `marker` was worse than wrong: it was being set to
  mean "keyframe", which `BufferFlags::SYNC_POINT` already says correctly two
  lines earlier, while the RTP marker bit actually means "last packet of an
  access unit" — so an RTP-aware consumer read the one field that lied. The RTP
  timestamp is not lost; `pts` is derived from it. `RtpSrc` is untouched: it
  reads real packets and populates `RtpMeta` honestly.

- **`EncoderStats` is no longer `Eq`** (#246). It gained three `f64` rate fields
  and an `EncodeLatency`, so the derive had to go. `PartialEq`, `Copy`, `Clone`,
  `Debug` and `Default` all remain, which is everything `assert_eq!` and the
  rest of the tree actually used — and asking whether two stats snapshots taken
  at different instants are *equal* was never a meaningful question. Matches
  `Queue2Stats`, which has carried `f64` rates beside `u64` counters all along.

**`MediaFormat` and encoded geometry** (#236). `VideoCodec` gains `Mjpeg`,
appended last so the discriminants `ipc_ring`'s `descriptor_enum_discriminants_locked`
pins do not move. `Metadata` gains a `coded` field carrying encoded video's
geometry, written by `set_coded_video_dims` and read by `coded_video_dims`.
`MediaFormat::Video(VideoCodec)` is unchanged: most of its uses are caps
declarations or producers that know the codec and nothing else, and
`MediaFormat::compatible` compares the whole payload, so a struct variant
would have made two H.264 caps at different resolutions incompatible.

`video_dims()` deliberately still answers only for raw video. Several readers
treat a `Some` from it as proof that a buffer has addressable planes.

Every video encoder now stamps what it produced instead of cloning its input's
`VideoRaw` — including `EncoderElement`'s flush path, which used to emit
`format: None` so the tail of a stream disagreed with its head. `VideoEncoder`
gains `codec()`; it is not part of the plugin ABI, so the ABI version is
unchanged.

**Zenoh attachment v1 → v2** (#236). `WireMetadata` carries the coded
geometry. Peers of different versions fall back to foreign-publisher handling,
as they always have.

**IPC descriptor semantics** (#236). `FMT_VIDEO` now uses the four payload
words `FMT_VIDEO_RAW` already spent, so the descriptor stays 128 bytes and
`IPC_RING_VERSION` does not move: an older peer never reads them and a newer
one sees zeros, so both directions degrade to "geometry unknown".

**`ZenohSink` builders return `Result`** (#231). `without_metadata()` and
`with_forward_custom_keys()` are meaningless on `@media` and now say so
instead of silently doing nothing.

**Plugin ABI 11 → 12** (#221). `AsyncElementDyn` gains `finish`, a terminal
hook the sink task awaits after the last buffer and before dropping the
element. New vtable slot, so every out-of-tree plugin must be rebuilt.

**Zenoh attachment v2 → v3** (#224). `WireMetadata` gains `terminal`. A
zero-payload sample carrying it is the parallax plane's end-of-stream marker;
it rides the media key expression, because zenoh orders only within a flow and
a terminal on a sibling key could overtake the last frame.

**Zenoh attachment v3 → v4, and the IPC descriptor 128 → 192 bytes** (#219).
One schema, one code table, two encodings — and both grew the three fields the
drift had eaten: `rtp` (carried by IPC, dropped by zenoh), `planes` (dropped by
both, so a strided producer's frame arrived claiming to be packed) and
`WireFormat::Rtp`. The descriptor takes `planes` inline rather than pushing it
through the control socket, which is what costs the extra 64 bytes; it is still
written scalar-by-scalar with nothing allocated per buffer.

`IPC_RING_VERSION` stays 2 — the *layout* around the descriptor did not change
— so note that the two version constants move independently and on purpose:
`WIRE_VERSION` is semantics, `IPC_RING_VERSION` is mechanics. Skew policy
differs by transport, also on purpose: zenoh treats an unknown version as a
foreign publisher and falls back, IPC gates on exact equality and errors,
because both ends of an IPC channel are parallax and a mismatch is a
deployment error rather than someone else's traffic.

**`RtspSrc` reconnects by default** (#206). A stream error used to end the run;
it now rebuilds the session against the same URL with exponential backoff and
full jitter, unlimited retries, posting a bus `Warning` per attempt and
stamping `DISCONT` on the first resumed buffer. That is right for a camera and
wrong for batch work, which wants a failure to be a failure — `.without_reconnect()`
or `reconnect=false` restores the old behaviour. EOS is never retried: a server
that said goodbye said goodbye.

**The RTP payloaders are `Transform`s** (#218). All six emit one buffer per RTP
packet via `Output::Multiple` instead of concatenating an access unit's packets
into one buffer for `RtpSink` to wrap in a single header — which produced a
malformed, oversized packet for every frame that fragmented.

**Demuxers stamp `Metadata.format`** (#238). MKV and MP4 also stamp `coded`
from the track header. Nothing was declared before, so `mp4demux ! zenohsink`
published H.264 as `zenoh/bytes`.

**`RtpH265Depay` emits NAL units** (#240). It used to hand back the RTP payload
verbatim — payload header attached, no start codes, fragmentation units never
reassembled. Anything consuming its output was consuming garbage.

**Element factory** (#222). `ElementConstructor` is an enum (`Sync`/`Async`)
rather than a bare fn-pointer alias, and `Props` is owning and cloneable
instead of borrowing, with its consumption set shared between clones. Both are
`pub`, though neither is referenced anywhere outside the factory module and
the plugin ABI does not use either — so no ABI bump. `register` is unchanged;
`register_async` is new.

### Added

- **Receive-side rates on `ZenohSrcHandle`** (#256). `samples_per_second()` and
  `bits_per_second()`, measured the way #246 measures the sink side — payload
  bits, same window — so the two are comparable across the wire and the gap
  between them is loss rather than an artefact of two definitions. The drop and
  loss counters stay counters: they mean deliberately different things and
  point at different machines, and a smoothed version of any of them says less.
- **CI compiles nine features and three crates it never did** (#255): `opus`,
  `av1-decode`, `av1-encode`, `eac3`, `audio-flac`, `audio-mp3`, `display-gpu`,
  `macros` and `cli` — the last of which gates `parallax-launch` and its test
  suite — plus `parallax-player`, `parallax-pipeline-macros` and
  `examples/example-plugin`, which no job built because none passes `-p`.
  Two new jobs (`codec-libs`, `workspace-members`) and two extensions.
  `default-members` is unchanged, so a bare `cargo build` is unaffected.

- **VP9 decodes into buffers parallax supplies** (#198). With
  `MemoryType::External` negotiated, `VpxDecoder` hands libvpx its own frame
  buffers through `vpx_codec_set_frame_buffer_functions` and emits the decoded
  frame in an `ExternalSlot` with libvpx's real strides — no plane copy at all,
  the shape `Dav1dDecoder` already used for its pictures. **VP9 only**: libvpx's
  header says the API "currently only works with VP9", so VP8 declares
  `cpu_only()` caps rather than advertising a capability it would silently fall
  back from on every frame. The pool's load-bearing rule is that a buffer has
  **two** independent holders — libvpx's release callback and the pipeline's
  last `Buffer` — and returns to the free list only when both are done; handing
  it back early would let the decoder draw the next frame over one still on
  screen. A frame whose planes do not satisfy `PlaneLayout::full_span_len`
  falls back to the packed copy with a warn-once, which is always legal because
  `external_preferred()` consumers also accept CPU memory.
  `VpxDecoder::pooled_frame_buffers()` reports the pool size.

- **Rates beside the counters, and an encode-latency distribution** (#246).
  Every control handle reported monotonic counters and not one rate, so every
  consumer wrote its own sampling loop with its own window — three
  implementations of one EWMA are three chances to disagree about what "encoded
  fps" means. `control::RateMeter` is now that one definition: a counter that
  also knows its own rate, folded over a stated `RATE_WINDOW` of 500 ms with a
  `RATE_TIME_CONSTANT` of 0.72 s, lock-free and allocation-free, and free on the
  hot path because the encoders already read the clock to time themselves.
  `EncoderStatsHandle` gains `frames_encoded_per_second`, `bits_per_second`
  (the same units as `EncoderControl::set_bitrate`, so the reading and the knob
  are directly comparable) and `rc_drops_per_second`; `ZenohSinkHandle` gains
  `samples_per_second` and `bits_per_second`. The counters are unchanged — a
  rate is what a human reads, a counter is what you difference and export.
  A stopped stream decays toward zero rather than freezing at its last value.
  `EncoderStatsHandle::encode_latency()` returns an `EncodeLatency`: p50/p95/p99
  plus min/max from a 768-byte lock-free histogram, four sub-buckets per octave
  so a percentile is located within 19% rather than within a factor of two.
  `last_encode_ns` stays — one sample cannot show a tail, and a tail is what
  causes a dropped frame. `docs/elements.md` § "Runtime control" has one table
  saying which figure is a counter, which is a rate, and over what window.

- **`rtspsrc`, `zenohsrc` and `zenohsink` from a pipeline string** (#222).
  `Pipeline::parse_async` / `parse_with_factory_async` await constructors that
  perform real I/O; `Pipeline::parse` stays and, on such an element, points at
  the async entry point rather than claiming a missing cargo feature.
  `parallax-launch` uses the async path, so

  ```
  parallax-launch "zenohsrc key=…/@media/parallax/cam0/video/h264/high profile=media ! h264dec ! autovideosink"
  ```

  works. Both zenoh elements share one process-wide session, opened on first
  use and configurable with `PARALLAX_ZENOH_CONFIG`.
- **Demand signalling on `ZenohSink`** (#234). `ZenohSinkHandle` (via
  `control()`, cloned before `start()`) reports `has_viewers()` plus monotonic
  rise/fall edge counters, delivers `ZenohDemandEvent` on a bounded lossy
  channel, exposes the publish counters, and carries `request_keyframe()`.
  A viewer arriving raises `Event::ForceKeyUnit` upstream automatically, so a
  late joiner does not wait out a GOP.

  There is deliberately **no viewer count**: zenoh's `MatchingStatus` is a
  boolean, and a count cannot be synthesised. On a build without
  `zenoh-unstable`, `has_viewers()` fails **open** so an idle reaper never
  kills a live stream because the build cannot tell.
- **The `@media` plane** (#231, #223). `ZenohSink::media` / `ZenohSrc::media`
  speak the ratified contract: one access unit per sample, codec on the
  `Encoding`, CBOR `FrameMeta` on the attachment, the `frame` QoS profile,
  exact keys only. The publisher owns the wire sequence and decides the
  keyframe flag from the bytes it is sending; the subscriber gates on the
  first keyframe and derives its caps from the key expression, so
  `prepare()` is deterministic.
- **`wire::FrameMeta`** (#232), with a CBOR conformance corpus under
  `tests/fixtures/framemeta/`.
- **`codec::annexb`**: `NalCodec`, `is_entry_point`, `has_param_sets`,
  `param_set_blob`, `ParamSetCache` (#233) and `h264_profile_level_id` +
  `h264_browser_compatible` (#215).
- **`Event::ForceKeyUnit`** (#213), the third upstream family. Consumed by
  the encoders, originated by `ZenohSink::request_keyframe`.
- **RTSP buffers describe themselves** (#205): codec from the SDP rtpmap name,
  geometry from `sprop-parameter-sets` / `a=framesize` / the first in-band SPS.
- **A terminal hook for sinks** (#221). `finish(EndReason)` on `Sink`,
  `AsyncSink`, `SimpleSink` and `AsyncElementDyn`, awaited from **all seven**
  exits the sink task has and bounded by `ExecutorConfig::finish_timeout`
  (default 2 s). `Mp4FileSink`, `HlsSink`, `DashSink` and `WebSocketSink` move
  their finalization out of `Drop`, where the error went nowhere.
- **End of stream across the wire** (#224). A terminal sample on the parallax
  plane — `Eos`, or `Error` with a message, so a stream that failed at the far
  end does not read as a clean end at this one — plus a **liveliness token on
  both planes**, which covers `kill -9` and network partition. `PublisherLoss`
  is `Wait` by default, because a camera pipeline restarting is normal
  operation. Also `ZenohSrcHandle::end_stream()` and `with_stream_timeout`.
- **`ExecutorConfig::live_video()`** (#237) and a "Stopping a live pipeline"
  section in `docs/pipeline.md`: `stopper()` outlives the handle `wait()`
  consumes, and no source wrapper is needed.
- **`codec::vpx`** (#239): `vp8_is_keyframe` / `vp9_is_keyframe`, the VPx half
  of `annexb::is_entry_point`.
- **`docs/multi-machine.md`** (#229), the deployment guide, and
  `examples/62_zenoh.rs` / `examples/63_media_consume.rs` (#228).
- **RTSP reconnection** (#206), on by default with exponential backoff and
  full jitter, `DISCONT` on the resumed stream and a re-armed keyframe gate.
  `without_reconnect()` restores fail-the-run for batch work.
- **RTCP Sender Reports** (#207): `StreamInfo::sender_report` and a PTS that
  is a wallclock rather than a per-session relative number, so two machines
  can agree on when a frame was captured. Plus `RtspUnknownRtcpSsrc`, because
  retina's default drops RTCP from an unexpected SSRC — and the anchor with it.
- **An RTSP test server** (#209) in `tests/support/`, hand rolled, whose
  failure modes are the point: a mid-stream disconnect, a stalled DESCRIBE, an
  SDP missing its parameter sets.
- **One interned custom-key registry** (#220) and **one wire schema** (#219),
  with `tests/wire_projection.rs` proving the IPC descriptor and the zenoh
  attachment agree.

### Fixed

- **`HlsSink::finish` never closed a live playlist.** `#EXT-X-ENDLIST` was
  written only when `is_vod`, so a live stream that ended left a playlist every
  player reads as still-running and keeps polling — the exact failure the
  `finish` hook's own doc comment describes. It is now written whenever the
  stream is over, which is what the tag means: no more segments are coming.
  Found by writing the first end-to-end test of #221's two real users; before
  it, the only `ENDLIST` assertion in the tree asserted its *absence*.
- **`vaapih265dec` handed out advice you could not act on.** It sat in
  `GATED_ELEMENTS`, so a build that already had `vaapi` was told to "rebuild
  with --features vaapi". HEVC is not gated by a feature — it is not built at
  any combination, because `cros-codecs`' `h265` feature does not compile
  against its published release (#200). A second category,
  `UNAVAILABLE_ELEMENTS`, now says that instead.
- **CI compiled four combos it never ran.** `display-check` and `vulkan-check`
  were clippy/check only, leaving 75 already-written test functions compiled and
  never executed — the display geometry and colour maths, the dma-buf plane
  validation, and the CPU-only half of the Vulkan decoder (slice headers, POC,
  reference marking). Both run `nextest` now. `image-png` was linted and
  documented but never tested, and `cli` and `zenoh` were never enabled
  together, so the three async-constructor acceptance tests for #222 had never
  run at all. Both are fixed in `feature-gates`.

- **`Queue2`'s rate estimation is `control::RateMeter`** (#257) instead of a
  second hand-rolled EWMA. One user-visible change: the buffering ETA now
  responds in about 0.72 s rather than 1.4 s, so it is a little jumpier and a
  lot more honest — the old estimator held its last value indefinitely, so a
  stalled download kept reporting the throughput it used to have.
- **The README's feature table** (#259) documented three features that had been
  deleted (`huge-pages`, `gpu`, `rdma`) and omitted six that exist (`cli`,
  `display-gpu`, `eac3`, `hotplug`, `v4l2-m2m`, `zenoh-unstable`). A test now
  diffs it against `[features]` in both directions.
- **`toolchain_pin` checked the action ref, not the compiler** (#254). A job
  written as `rust-toolchain@master` with `toolchain: stable` floated exactly
  as badly as a floating action ref and passed the check meant to stop it.

- **`tests/zenoh_media.rs` asserted delivery on a plane that promises none.**
  Two separate failures, one root cause each. `settle()` slept a fixed 300 ms
  and hoped the subscription had propagated; it now declares a throwaway probe
  subscriber and publishes to it until something arrives, which *proves*
  propagation — sound because the probe is declared after whatever the test
  declared and both travel the same link, which delivers declarations in order.
  It is also faster than the sleep it replaced. Separately,
  `the_gate_holds_until_the_first_keyframe` published one burst and required all
  of it to arrive: instrumenting a starved run showed the sink reporting four
  samples sent against two received with `lost_wire == 0` — no wire gap, just
  samples `CongestionControl::Drop` was entitled to discard. It now republishes
  until a round lands, and only checks the pre-keyframe counter when a delta
  actually arrived. Verified 25/25 and the whole binary 5/5 under twenty
  spinning threads pinned to two cores, where it previously failed one run in
  six.

- **`just check-media-full` was red on master** and had been since the 1.98
  toolchain pin. `f2625b1` swept clippy's new `drain_collect` out of the AAC
  encoder but missed its twin in the Opus one, because no CI job builds `opus` —
  the feature appears only in that recipe, which no workflow mirrors. It is also
  the only recipe that covers `vpx`, so the gap hid a second thing.

- **`a_stalled_consumer_bounds_memory_instead_of_growing_it` was flaky, and had
  turned CI red on master.** It published one fixed burst 300 ms after
  declaring the subscriber, then gave the callback one second to show an
  eviction. Under CPU contention the subscription had not propagated yet, so
  the whole burst went into the unmatched window and vanished — `@media` is
  best-effort and drop-by-contract — leaving `samples_received == 0` and a
  failure that said nothing about the bound under test. It now republishes
  until the queue visibly evicts, and its assertion prints
  `samples_received` so a future failure distinguishes a lost handshake from a
  real regression. Verified at 10/10 under 24 spinning threads on a 6-core
  box, still finishing in under a second.

- **Two `ZenohSinkHandle` counters had no writer at all** (#246).
  `keyframes_sent` and `publish_errors` were declared, exposed on the handle and
  never incremented by anything; `tests/zenoh_media.rs` asserted
  `publish_errors() == 0` and passed vacuously. Both are wired now. Note what
  `publish_errors` does and does not mean: a failed publish is fatal to the sink
  task, so it is bounded by one, and it is **not** a congestion signal — under
  `CongestionControl::Drop`, which the `@media` profile pins, a congested link
  drops the sample and returns success, and zenoh offers no way to read that
  back. That is why neither counter has a rate.

- **Every `@media` sample parallax published was unstamped** (#245). zenoh
  enables session timestamping for *routers* by default and parallax opens as a
  peer, so its publishers ran without an HLC. `FrameMeta` carries no wallclock
  on purpose, which makes that stamp a consumer's only clock for frame age — and
  the failure was silent in the worst way: a receiving session with timestamping
  on stamps unstamped incoming samples itself, so consumers read their own clock
  as the publisher's and saw a frame age of zero. Every session parallax opens
  now enables `timestamping/enabled` (`network::session_config`, exported beside
  `shared_session`). An operator's `PARALLAX_ZENOH_CONFIG` still wins: a file
  that turns timestamping off keeps it and gets one warning, while a file that
  says nothing about it gets the default filled in.

- **`ZenohSrc`'s receive queue was unbounded** (#225). A pipeline that stopped
  pulling grew memory without limit. It is now a bounded drop-oldest queue
  (8 samples — deep queues turn publisher-side loss into latency, which is the
  wrong trade for live media), with three *separate* drop counters, because
  they point at different machines: `dropped_queue` (this host is too slow),
  `dropped_no_slot` (this pipeline's arena is undersized), `lost_wire` (the
  samples never arrived). Exposed through `ZenohSrcHandle` via `Controllable`.
- **A deliberately dropped sample cost 1 ms** (#225). The keyframe gate and
  arena exhaustion both returned `Ok(None)`, which `produce` mapped to
  `WouldBlock`, which the executor answers with a 1 ms sleep — so a pre-keyframe
  burst was paced at 1 kHz. They are now distinguishable, and only real
  back-pressure yields.
- **`ZenohSrc` did not hold its session alive**. It declared a subscriber and
  dropped the `Session`; dropping the last clone closes the session and makes
  `produce` report EOS on its first poll. Invisible until a test stopped
  sharing one process-wide session.
- **RTP keyframe detection** (#239). H.264 read only the first NAL, so the
  canonical `SPS | PPS | IDR` keyframe was delivered *unflagged*; H.265 and VP9
  had no detection at all and inherited a `SYNC_POINT` that `RtpSrc` sets from
  the RTP **marker** bit — "end of access unit", not "keyframe" — so roughly
  every frame claimed to be an entry point. Every verdict now sets *and clears*
  from the bitstream.
- **Three payload copies per zenoh buffer, down to one** (#226). Publishing
  hands zenoh the arena's own bytes; receiving walks `ZBytes::slices()` straight
  into a slot. In-flight publishes therefore pin producer slots, which
  `ZenohSink::retained_buffers()` declares — on the parallax plane that turns
  unbounded heap growth under `CongestionControl::Block` into real backpressure.
- **`@media` no longer emits a zero-length "heartbeat" frame** (#224).
  `with_timeout` on that plane produced an empty `TIMEOUT`-flagged buffer, which
  a decoder cannot tell apart from a malformed access unit.
- **A load-dependent test flake** (#241). `monitored_link_reports_busy_then_ready`
  bounded its wait with 200 000 `yield_now()` calls — a CPU budget, not a
  timeout, and one that shrinks on a faster box while the thing it waits for
  gets slower under load.

## [0.7.0] — 2026-08-19

175 commits since 0.6.0. Every out-of-tree plugin must be rebuilt: the plugin
ABI went **1 → 11** and the shared-arena format **v4 → v5**.

### Breaking

**Plugin ABI 1 → 11.** Element instances cross the boundary as double-boxed
`DynAsyncElement`, so any method added to `AsyncElementDyn` changes the vtable.
Ten bumps this cycle: `set_output_budget` (#84), `process_demux` (#76),
`EventResult::Handled` gaining a landing position, `EventResult::Forward` +
`seek_translations`, the inline fast-path slots (#175), `take_upstream_event`
(#173), `latency` (#184), `set_negotiated_memory` (#145), `retained_buffers` +
`passthrough` (#189), and `MemoryType::External` (#194).

**Arena format v4 → v5** (#177). `validate()` rejects v4. The byte layout is
unchanged; the bump guards the *access pattern* — each `SlotHeader` is now one
`AtomicU64` acquired with a single CAS, and a v4 peer CASing the state
half-word alone would race the v5 orphan sweep back into double allocation.

**IPC wire format** (#179). `ControlMessage` is incompatible with pre-#179
peers. The per-buffer path moved to a shared-memory descriptor ring with
eventfd doorbells; the socket now carries only registration and shutdown.

**Memory**
- `MemoryHandle` is a `#[non_exhaustive]` enum — `Cpu` | `DmaBuf` |
  `External` (#145, #194). `new`/`with_len` still build `Cpu`. `as_mut_slice`,
  `slot`, `ipc_ref` and the `arena_*` accessors return `Option`;
  `as_bytes_mut` panics on a read-only mapping, `try_as_bytes_mut` is the
  checked form.
- `ProduceResult::OwnDmaBuf` and `DmaBufBuffer` are gone — a dmabuf frame is
  `OwnBuffer` carrying `MemoryHandle::DmaBuf`, and `Buffer::copy_to_cpu`
  replaces `DmaBufBuffer::to_buffer`.
- `EventFd` moved from `pipeline::rt_bridge` to `memory` (#180).
- `MemorySegment`, `IpcHandle`, `MappedFileSegment`, `HugePageSegment`,
  `AtomicBitmap`, `MemorySrc`, `MemorySink`, `SharedMemorySink` deleted, along
  with the `huge-pages` and `rdma` features — nothing consumed them (#186).

**Pipeline**
- The `Queue` element is gone: the link *is* the queue (#167). `FlowPolicy`,
  `FlowStats` and `Source::handle_flow_signal` went with it, and `FlowSignal`
  narrows to `Ready | Busy`.
- `LinkPolicy::Drop` is now `DropNewest`, with no deprecated alias, and
  `DropOldest` joins it (#169).
- Fan-in into anything that is not a muxer is a **link-time error** (#183).
  Graphs that linked two producers into one sink used to compile and silently
  drop one of them.
- `parallax::link` deleted — `IpcPublisher`, `IpcSubscriber`, `NetworkSender`,
  `NetworkReceiver`. Use the IPC and network *elements* (#179).

**Elements**
- `AppSink` is an `AsyncSink` only: `add_sink` becomes `add_async_sink`
  (#168). `IpcSrc` likewise becomes `add_async_source` (#179), and
  `IpcSink::with_max_pending` is `with_capacity`.
- The `VideoDecoder` trait and `DecoderElement` are gone; video decoders
  implement `Element` directly (#160). The legacy `"width"`/`"height"` custom
  metadata keys carry no geometry any more — `Metadata::set_video_dims` and
  `video_dims()` are the representation.
- `Gain::process` requires `MediaFormat::AudioRaw` on every buffer and errors
  without it, at any factor including unity; mute writes per-format digital
  silence.
- Pull results are a `Pulled { Buffer, Empty, Flushing, Ended(EndReason) }`
  rather than a `Result`; `is_eos()` is replaced by `end_reason()`/`ended()`
  (#85).
- `Error::Shutdown` and `Error::Panic` added. `Shutdown` is a cooperative
  quit ending the run as `EndReason::Eos` — `AutoVideoSink` returns it on
  window close, so string-matching `"Display window closed"` must stop
  (#191, #85).
- `SeekableSource`, `SeekRequest`, `Event::LatencyQuery` removed;
  `MessageKind::Qos` redesigned to match the new `QosEvent` (#184).
- MP4 `seek_to_time`/`seek_all_to_time` take a `SeekSnap` (#166).
- `kanal` dropped for `tokio::sync::mpsc`; `MetadataExtract::new` returns a
  tokio `UnboundedReceiver`.

**VA-API** (new this cycle, so this is scoping rather than a break): HEVC
hardware decode is **not built**. `cros-codecs`' `h265` feature does not
compile against its published release, and a driver without an H.264 decode
config gets no VA-API at all, because the dependency initialises every backend
with a hardcoded H.264 config and panics without one. Both are upstream
defects tracked in #200; `[patch.crates-io]` used to hide them from us alone
and was removed, since cargo strips it from the published manifest (#202).

### Added

- **VA-API hardware decode** (#193) — H.264, VP8 and VP9 on the GPU video
  engine, each verified bit-exactly against its software counterpart, with the
  decoded dma-buf handed to the display instead of copied (#62). 1080p H.264:
  0.54 → 0.12 CPU cores, arena 88 MB → 34 MB.
- **`parallax-launch`** (#187, feature `cli`) — the gst-launch equivalent,
  plus factory registration for ~65 element names.
- **GPU presentation** (#190, feature `display-gpu`) — wgpu backend taking
  I420/NV12 and doing colorspace conversion and letterbox scaling in a shader.
- **`parallax-player`** — a real player over the engine: A/V sync, pause,
  seek, position/duration, volume, container and codec dispatch.
- **Seeking, end to end** (#162–#166, #173, #183) — seekability queries and
  gating, hop-by-hop upstream dispatch, seek translation for push-mode
  demuxers, `SNAP_BEFORE`/`SNAP_AFTER`, `ACCURATE` with iterative refinement,
  SEGMENT seeks with `SegmentDone`, keyframe-only fast-forward and reverse,
  instant rate change, and segment discipline throughout.
- **Zero-copy memory** — `MemoryType::External` with per-buffer `PlaneLayout`
  so producers keep their own strided frames (#194), strided input across the
  converters, scaler and encoders (#196), dmabuf flow-through (#145),
  executor-sized output arenas with admission control (#84, #91, #189), a
  release-queue doorbell (#180) and orphaned-slot recovery (#177).
- **Cross-process IPC on a descriptor ring** (#179) — zero-alloc hot path,
  eventfd doorbells, start-order independence.
- **`HttpCacheSrc`** (#188) — ranged HTTP download fused with a sparse
  write-through cache, so a backward seek into a downloaded span is served
  from disk. Queue2 gained buffering-aware seeking and downloaded ranges
  (#164).
- **QoS and latency** (#184) — sink-originated `Event::Qos` travelling
  upstream and mirrored to the bus, plus static latency aggregation.
- **Runtime pause/resume/position** (#71) and terminal-outcome reporting via
  `PipelineHandle::ended()`.
- New codecs and containers: VP8/VP9 (`vpx`), Vorbis, AAC and AC-3/E-AC-3
  (`eac3`) decoders, Matroska/WebM demux (`mkv-demux`), AV1 and Opus tracks in
  MP4, audio downmix.

### Fixed

- A fatal element error now winds the whole run down instead of letting
  siblings drain to EOS (#191).
- A panicking probe or tracer callback is caught, logged and unregistered
  rather than killing the pipeline (#92).
- Adapters drain the author `flush()` until `None`, so EOS no longer truncates
  held frames.
- `PipeWireSrc`/`PipeWireSink` never ran at all — an inverted shutdown check.
- `aac-encode` did not compile, against any published `fdk-aac`, and no CI job
  or `just` recipe built it. Fixed, and `aac-encode`, `websocket`, `alsa` and
  `v4l2-m2m` now have a CI job of their own.
- `screen-capture` ported to ashpd 0.13.

## [0.6.0] — 2026-08-08

Breaking: `GpuMemory::free` removed (GPU buffers free on `Drop` — a manual
free would double-free); `HwVideoDecoder` gained a required `read_frame`;
`HwDecoderElement::with_dimensions` removed per the geometry-in-Metadata
invariant; `h264_std::std_sps`/`std_pps` return the scaling lists alongside
the struct.

Also: `HwDecoderElement` bridges decoded pixels into pipeline buffers (#57),
`Mp4Demux` emits decodable Annex-B with in-band SPS/PPS (#58), `frame_num`
gaps error-then-skip-until-IDR instead of decoding corrupt (#59),
`AppSrcHandle::try_push_buffer` (#60), H.264 scaling matrices are marshalled
rather than rejected (#61).

0.4.0 and 0.5.0 were never published to crates.io, so 0.6.0 carried
everything since 0.3.0.

[0.7.0]: https://github.com/p13marc/parallax/releases/tag/0.7.0
[0.6.0]: https://github.com/p13marc/parallax/releases/tag/0.6.0
