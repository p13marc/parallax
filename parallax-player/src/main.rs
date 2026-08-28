//! A video player built on parallax and Iced.
//!
//! The engine does the hard parts — demux, decode, A/V sync, seeking — and
//! this is the front end: a window, a video surface, and the transport.
//!
//! ```text
//! demux ─┬─ video ─▶ decode ─▶ AppVideoSink ⇄ mailbox ⇄ VideoView (this)
//!        └─ audio ─▶ decode ─▶ downmix ─▶ gain ─▶ AlsaSink (the clock)
//! ```

mod dialog;
mod engine;
mod headless;
mod playlist;
mod recent;
mod snapshot;
mod thumbnail;
mod ui;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use iced::keyboard::{self, Key, key::Named};
use iced::widget::{center, container, stack};
use iced::{Element, Subscription, Task, Theme};

use parallax::clock::ClockTime;
use parallax::event::{SeekEvent, SeekFlags};
use parallax::pipeline::MessageKind;
use parallax::video::FrameMailbox;

use engine::{Engine, HwDec, Options, Probed, SourceDescription};
use playlist::Playlist;

/// Long enough to swallow a held arrow key's repeats, short enough that a
/// single press still feels immediate.
const SEEK_DEBOUNCE: Duration = Duration::from_millis(120);

/// How long a newer target waits for the outstanding seek's `SeekDone` before
/// going out anyway.
///
/// `SeekDone` normally lands within a frame, so this only fires when one is
/// lost or a demuxer is genuinely stuck — and then a superseding seek is the
/// right answer, not an unresponsive transport.
const SEEK_SUPERSEDE: Duration = Duration::from_secs(1);

/// How often the interface re-reads position, drops and the bus.
const TICK: Duration = Duration::from_millis(50);

/// How long the controls stay up after the last thing the user did.
const CONTROLS_LINGER: Duration = Duration::from_secs(3);

/// The speeds `s` cycles through, and the buttons the transport offers.
///
/// No reverse: an `INSTANT_RATE_CHANGE` cannot flip the sign — that needs a
/// real seek, and no demuxer in the tree plays backwards.
const SPEEDS: [f64; 5] = [1.0, 1.25, 1.5, 2.0, 0.5];

/// How long a passing notice stays on screen.
///
/// Long enough to read a sentence, short enough that a stream posting one a
/// second does not bury the picture.
const NOTICE_LINGER: Duration = Duration::from_secs(8);

/// How many notices are kept at once. The oldest goes first.
const NOTICE_LIMIT: usize = 4;

/// How bad the news is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Something happened worth saying once.
    Info,
    /// Something is wrong but playback continues.
    Warning,
    /// The run is over.
    Error,
}

/// One thing the pipeline said, with the time it said it.
#[derive(Debug, Clone)]
pub struct Notice {
    text: String,
    level: Level,
    at: Instant,
}

/// Whether a notice has been on screen long enough.
///
/// Errors never expire: the run is over, and a message that vanishes before
/// it is read is the same as no message. Everything else is passing news, and
/// the picture is what the user came for.
fn notice_expired(level: Level, age: Duration) -> bool {
    level != Level::Error && age >= NOTICE_LINGER
}

/// One request to put a file on screen.
///
/// The seqnum is what lets a probe that answers a superseded request be
/// dropped rather than built: an Iced `Task` cannot be cancelled, so the
/// answer always arrives and something has to decide it is stale.
#[derive(Debug, Clone)]
struct Request {
    seq: u64,
    index: usize,
    path: PathBuf,
}

/// Where the open chain is.
///
/// The invariant the type exists to hold: an `Engine` exists only in
/// `Playing`, exactly one teardown is in flight in `TearingDown`, and exactly
/// one probe is outstanding in `Probing`.
#[derive(Debug, Default)]
enum Stage {
    /// Nothing open.
    #[default]
    Idle,
    /// The previous pipeline is winding down; this is what follows it.
    TearingDown { next: Option<Request> },
    /// A file is being read for what it contains.
    Probing(Request),
    /// An engine is running.
    Playing,
}

/// What opening a file does next, given where the chain already is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    /// Wind the running pipeline down first.
    Teardown,
    /// Nothing is running; read the file.
    Probe,
    /// A teardown is already in flight — wait for it, and let it pick this
    /// up. Spawning a second one is how two pipelines end up alive at once.
    Wait,
}

impl Stage {
    /// What to do about a request arriving here.
    fn open(&self) -> Open {
        match self {
            Self::Playing => Open::Teardown,
            Self::TearingDown { .. } => Open::Wait,
            Self::Probing(_) | Self::Idle => Open::Probe,
        }
    }

    /// Whether a probe answering `seq` is still the one being waited for.
    fn accepts_probe(&self, seq: u64) -> bool {
        matches!(self, Self::Probing(request) if request.seq == seq)
    }
}

/// Where a step of `frames` from `from` lands, in nanoseconds.
///
/// Clamped at both ends, and the far end matters: a duration of `NONE` means
/// *unknown* — an MKV with no Segment Duration — not an empty file, so it
/// must not clamp to zero. That is the same trap `seek_relative` documents.
fn step_target(from: i64, frame: Duration, frames: i32, duration: ClockTime) -> u64 {
    let step = frame.as_nanos() as i64 * i64::from(frames);
    let mut target = from.saturating_add(step).max(0);
    if let Some(total) = duration.to_option() {
        target = target.min(total.nanos().saturating_sub(1) as i64);
    }
    target as u64
}

/// How far one scroll gesture moves the volume, in dB.
///
/// A notch of a wheel is one step of the arrow keys, so the two agree. A
/// touchpad reports pixels instead, and its own idea of how many make a
/// notch, so those are scaled and clamped: an inertial flick should not swing
/// the volume through its whole range.
fn volume_step(delta: iced::mouse::ScrollDelta) -> f32 {
    const PIXELS_PER_NOTCH: f32 = 50.0;
    const STEP_DB: f32 = 2.0;
    let notches = match delta {
        iced::mouse::ScrollDelta::Lines { y, .. } => y,
        iced::mouse::ScrollDelta::Pixels { y, .. } => y / PIXELS_PER_NOTCH,
    };
    (notches * STEP_DB).clamp(-STEP_DB * 3.0, STEP_DB * 3.0)
}

/// Whether to stop skipping past files that will not open.
///
/// A full lap of the list with nothing playable is the end of it; anything
/// less keeps going, because one bad file in a directory is normal.
fn give_up(failures: usize, entries: usize) -> bool {
    entries > 1 && failures >= entries
}

/// A probe result exactly one reader may take.
///
/// An Iced `Message` must be `Clone`, and a probe result is not: it owns an
/// open demuxer. A cloned message therefore finds the cell empty and does
/// nothing, which is the right answer — the alternative, `Arc::try_unwrap`,
/// turns a harmless duplicate into a failed open.
#[derive(Debug, Clone)]
struct ProbeCell(Arc<std::sync::Mutex<Option<Result<Probed, String>>>>);

impl ProbeCell {
    fn new(result: Result<Probed, String>) -> Self {
        Self(Arc::new(std::sync::Mutex::new(Some(result))))
    }

    fn take(&self) -> Option<Result<Probed, String>> {
        self.0.lock().ok()?.take()
    }
}

/// Read a file for what it contains, off the interface's thread.
///
/// An MP4 index parse on a large file is 100 ms and more.
fn probe(request: Request) -> Task<Message> {
    let seq = request.seq;
    Task::perform(
        async move {
            tokio::task::spawn_blocking(move || {
                engine::probe(&request.path).map_err(|e| format!("{e:#}"))
            })
            .await
            .unwrap_or_else(|e| Err(format!("probe panicked: {e}")))
        },
        move |result| Message::Probed(seq, ProbeCell::new(result)),
    )
}

/// What the pipeline last said about itself, for the diagnostics panel.
///
/// All of it arrives on the bus and all of it used to land in `_ => {}`.
#[derive(Debug, Default)]
struct Diagnostics {
    /// Network buffering, 0-100.
    buffering: Option<u32>,
    /// Downloaded bytes and the total, when a download-mode queue says.
    downloaded: Option<(u64, Option<u64>)>,
    /// The last QoS advisory: required-vs-achieved rate, and worst lateness.
    qos: Option<(f64, i64)>,
    /// The clock the pipeline elected.
    clock: Option<String>,
    /// Iced's adapter and backend — the only honest answer to "which GPU is
    /// drawing", since `wgpu::Device` cannot name its own adapter and the
    /// custom-primitive path is handed nothing else.
    graphics: Option<(String, String)>,
}

/// The one machine-readable summary line, printed by both hosts.
///
/// `position_ns` and `dropped` stay adjacent and in that order:
/// `scripts/decode_matrix.py` parses exactly that pair, and #270 extends the
/// same script rather than replacing it. One function rather than two format
/// strings, because a windowed run and a headless one that disagreed about
/// the shape of this line would make the matrix silently unreadable.
fn stats_line(position: ClockTime, dropped: u64, stats: &parallax::video::FrameStats) -> String {
    format!(
        "stats: position_ns={} dropped={} presented={} queued={} late={} full={} stale={} clipped={}",
        position.to_option().map(|p| p.nanos()).unwrap_or(0),
        dropped,
        stats.presented,
        stats.queued,
        stats.dropped_late,
        stats.dropped_full,
        stats.dropped_stale,
        stats.clipped,
    )
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "parallax-player",
    about = "A video player built on parallax and Iced"
)]
struct Args {
    /// MP4/MOV or MKV/WebM files to play, in order.
    files: Vec<PathBuf>,

    /// Play without audio.
    #[arg(long)]
    no_audio: bool,

    /// Loop the file, gaplessly, inside the demuxer.
    ///
    /// `--loop` is kept as an alias: it is what the crossterm player called
    /// this, and scripts outlive rewrites.
    #[arg(long = "repeat", alias = "loop")]
    repeat: bool,

    /// Whether to try the GPU video engine.
    #[arg(long, value_enum, default_value_t = HwDec::Auto)]
    hwdec: HwDec,

    /// Finished frames the decoder may run ahead of the screen.
    #[arg(long, default_value_t = 4)]
    decode_ahead: usize,

    /// dav1d worker threads; 0 picks by resolution.
    #[arg(long, default_value_t = 0)]
    dav1d_threads: u32,

    /// dav1d frame-delay bound; 0 leaves the decoder's own choice.
    #[arg(long, default_value_t = 0)]
    frame_delay: u32,

    /// Executor link capacity for edges that do not set their own.
    #[arg(long, default_value_t = 4)]
    channel_capacity: usize,

    /// Decode as fast as possible instead of presenting at the due time.
    ///
    /// The batch-decode floor: it measures what the decode path costs with
    /// pacing out of the way, which is how #192 was measured. Playback is
    /// meaningless in this mode.
    #[arg(long)]
    no_sync: bool,

    /// ALSA device to play through; `--audio-device list` prints them.
    #[arg(long, default_value = "default")]
    audio_device: String,

    /// Video track to play, by container id; the default is the container's.
    #[arg(long)]
    video_track: Option<u64>,

    /// Audio track to play, by container id.
    #[arg(long)]
    audio_track: Option<u64>,

    /// Subtitle track to show, by container id; none by default.
    #[arg(long)]
    subtitle_track: Option<u64>,

    /// A SubRip file to show alongside the video.
    #[arg(long = "sub", value_name = "FILE.srt")]
    sidecar: Option<PathBuf>,

    /// Play with no window at all, printing the machine-readable summary.
    ///
    /// The pipeline is the same one the window drives, and this process is
    /// the host: it drains the frame mailbox at each frame's due time exactly
    /// as the widget does. What it cannot exercise is the shader and the
    /// swapchain — `just play-headless` runs the real widget under weston for
    /// that.
    #[arg(long)]
    headless: bool,

    /// Quit after this many seconds, printing a machine-readable summary.
    ///
    /// The window still opens and the pipeline still runs — this only bounds
    /// the run, so a headless box can prove playback works and measure it.
    #[arg(long)]
    exit_after: Option<u64>,
}

impl Args {
    fn options(&self) -> Options {
        Options {
            hwdec: self.hwdec,
            audio: !self.no_audio,
            gapless_loop: self.repeat,
            decode_ahead: self.decode_ahead,
            dav1d_threads: self.dav1d_threads,
            frame_delay: self.frame_delay,
            channel_capacity: self.channel_capacity.max(1),
            sync: !self.no_sync,
            audio_device: self.audio_device.clone(),
            video_track: self.video_track,
            audio_track: self.audio_track,
            subtitle_track: self.subtitle_track,
            sidecar: self.sidecar.clone(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Logs go to stderr so that stdout carries the `stats:` line and nothing
    // else: `scripts/decode_matrix.py` parses that stream, and
    // `tracing_subscriber::fmt()` writes to stdout by default.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    if args.audio_device == "list" {
        for device in parallax::elements::AlsaSink::enumerate_devices()? {
            println!("{}\t{}", device.name, device.description);
        }
        return Ok(());
    }

    if args.headless {
        let code = headless::run(args)?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }

    // `view` goes through a closure: as a bare method reference its elided
    // lifetime does not satisfy Iced's higher-ranked `ViewFn` bound.
    iced::application(
        move || Player::boot(args.clone()),
        Player::update,
        Player::view,
    )
    .subscription(Player::subscription)
    .title(Player::title)
    // The argument type is annotated because Iced's `ThemeFn` is
    // higher-ranked over the state's lifetime and cannot infer it from `_`.
    .theme(|_: &Player| Theme::Dark)
    .window_size((1280.0, 720.0))
    .run()?;
    Ok(())
}

#[derive(Debug, Clone)]
enum Message {
    /// The periodic re-read of position, drops and the bus.
    Tick,
    /// A frame landed on an idle mailbox.
    Frame,
    /// A file finished probing, answering this request.
    Probed(u64, ProbeCell),
    /// The previous engine finished tearing down; start whatever waits.
    TornDown,
    /// Add files to the playlist, playing the first if nothing is.
    Enqueue(Vec<PathBuf>),
    /// Play the playlist entry at this index.
    PlayIndex(usize),
    /// Drop the playlist entry at this index.
    Remove(usize),
    /// Play the next file.
    Next,
    /// Play the previous file.
    Previous,
    /// Step through off / all / one.
    CycleRepeat,
    /// Show or hide the playlist.
    TogglePlaylist,
    /// Play through this ALSA device from the next build on.
    AudioDevice(String),
    /// Play this track of the current file instead.
    SelectTrack(engine::TrackKind, u64),
    /// Play/pause.
    TogglePause,
    /// Seek by a number of seconds, relative to where the transport is headed.
    SeekBy(i64),
    /// Step by a number of frames, pausing first.
    StepBy(i32),
    /// Seek to a fraction of the duration — the scrubber.
    SeekTo(f32),
    /// Set the volume in dB — the slider.
    VolumeTo(f32),
    /// The pointer moved or a key was pressed: keep the controls up.
    Activity,
    /// Step the volume in dB.
    VolumeBy(f32),
    /// Mute/unmute.
    ToggleMute,
    /// Enter or leave fullscreen.
    ToggleFullscreen,
    /// Leave fullscreen, or quit.
    Escape,
    /// Ask the desktop for files to play.
    OpenDialog,
    /// The desktop could not be asked, and why.
    DialogFailed(String),
    /// Show or hide the diagnostics panel.
    ToggleStats,
    /// Write the running graph out as GraphViz.
    DumpGraph,
    /// Save the frame on screen as a PNG.
    Snapshot,
    /// Grow or shrink the subtitles.
    SubtitleSize(f32),
    /// A snapshot finished, for better or worse.
    Snapshotted(Result<PathBuf, String>),
    /// Dismiss the notice at this index.
    Dismiss(usize),
    /// What Iced's renderer is running on.
    Graphics(Box<iced::system::Information>),
    /// The cursor moved over the seek bar, `x` pixels along it (#282).
    ScrubHover(f32),
    /// The cursor left the seek bar.
    ScrubLeave,
    /// The seek bar reported its own size, which is the only way an x offset
    /// becomes a fraction of the file.
    ScrubberSized(iced::Size),
    /// A preview finished decoding: generation, instant, and whether the
    /// thumbnailer was already open when it started.
    Previewed(u64, u64, bool, Result<thumbnail::Thumb, String>),
    /// Step through the playback speeds.
    CycleSpeed,
    /// Set the playback speed directly.
    SpeedTo(f64),
    /// Quit.
    Quit,
}

/// Where a seek is, between the keypress and the `SeekDone`.
#[derive(Debug, Default)]
struct SeekState {
    /// Where the transport is headed, accumulated across presses.
    ///
    /// Not read back from `position()`: until the seek's `Segment` lands that
    /// still reports the *old* position, so a second press would walk from
    /// there and undo the first.
    pending: Option<u64>,
    /// When the debounce expires.
    due: Option<Instant>,
    /// The seek that has been dispatched: `(seqnum, target)`.
    in_flight: Option<(u64, u64)>,
    /// When it went out, for the supersede timeout.
    sent: Option<Instant>,
    /// Whether the pending seek must land on the frame asked for (#281).
    ///
    /// A scrub wants the nearest keyframe — it is faster and nobody can see
    /// the difference at scrubbing speed. A frame step wants the frame it
    /// named, which is what `ACCURATE` buys: the segment starts at the
    /// request while data starts at the keyframe, and the decoder
    /// decodes-but-drops the gap.
    accurate: bool,
}

impl SeekState {
    /// Where the transport is headed, if anywhere.
    fn target(&self) -> Option<u64> {
        self.pending.or(self.in_flight.map(|(_, t)| t))
    }
}

struct Player {
    options: Options,
    engine: Option<Engine>,
    /// Where the open chain is. `Engine` exists only in `Stage::Playing`.
    stage: Stage,
    /// Monotonic request number, so a stale probe can be recognised.
    seq: u64,
    /// What to play, and what to play next.
    playlist: Playlist,
    /// Consecutive files that failed to open, reset when one actually plays.
    failures: usize,
    /// Whether the playlist panel is on screen.
    show_playlist: bool,
    /// Files opened lately, newest first — the empty screen's only content.
    recent: Vec<PathBuf>,
    /// The cues on screen right now, re-read on the tick (#279).
    ///
    /// Polled rather than pushed: `VideoView::update` deliberately never
    /// publishes a message — Iced rebuilds the whole interface when a redraw
    /// produces one — and a cue boundary is comfortably inside the 50 ms
    /// tick's tolerance.
    cues: Vec<parallax::subtitle::Cue>,
    /// How large to draw them.
    subtitle_size: f32,
    /// Output devices, read once when someone first asks to see them.
    audio_devices: Vec<parallax::elements::AlsaDeviceInfo>,
    /// Where to resume after a rebuild that was not a track change.
    ///
    /// Changing the output device means building a new pipeline, and a
    /// pipeline that restarted from the beginning would be a worse answer
    /// than not offering the choice.
    resume_at: Option<u64>,

    position: ClockTime,
    paused: bool,
    /// Playback speed (#280). Audio is muted away from 1x — see
    /// [`Player::set_speed`].
    speed: f64,
    seek: SeekState,

    /// Volume lives here rather than in the `Gain` handle: the handle knows a
    /// linear factor, the interface thinks in dB, and a mute has to remember
    /// the level it will restore.
    volume_db: f32,
    muted: bool,

    fullscreen: bool,
    show_stats: bool,
    /// When the user last did anything, for the auto-hiding controls.
    last_activity: Instant,
    status: String,
    /// What the pipeline has said lately, newest last.
    notices: VecDeque<Notice>,
    /// What it has said about itself.
    diagnostics: Diagnostics,

    /// Seek-bar previews: the second decoder, its cache, and what is on
    /// screen (#282).
    previews: thumbnail::Previews,

    /// When to quit, for `--exit-after`.
    deadline: Option<Instant>,
}

impl Player {
    fn boot(args: Args) -> (Self, Task<Message>) {
        let player = Self {
            options: args.options(),
            engine: None,
            stage: Stage::Idle,
            seq: 0,
            playlist: Playlist::default(),
            failures: 0,
            show_playlist: false,
            recent: recent::load(),
            cues: Vec::new(),
            subtitle_size: 22.0,
            audio_devices: Vec::new(),
            resume_at: None,
            position: ClockTime::ZERO,
            paused: false,
            speed: 1.0,
            seek: SeekState::default(),
            volume_db: 0.0,
            muted: false,
            fullscreen: false,
            show_stats: false,
            last_activity: Instant::now(),
            status: "no file open".into(),
            notices: VecDeque::new(),
            diagnostics: Diagnostics::default(),
            previews: thumbnail::Previews::default(),
            deadline: args
                .exit_after
                .map(|s| Instant::now() + Duration::from_secs(s)),
        };
        let task = if args.files.is_empty() {
            Task::none()
        } else {
            Task::done(Message::Enqueue(args.files))
        };
        (player, task)
    }

    fn title(&self) -> String {
        match &self.engine {
            Some(engine) => format!("{} — parallax", engine.source.summary()),
            None => "parallax".into(),
        }
    }

    /// The running pipeline, for the diagnostics panel.
    fn engine(&self) -> Option<&Engine> {
        self.engine.as_ref()
    }

    fn has_audio(&self) -> bool {
        self.engine.as_ref().is_some_and(|e| e.has_audio())
    }

    /// Keep the controls up for a while longer.
    fn activity(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Whether the transport is on screen.
    fn controls_visible(&self) -> bool {
        controls_visible(
            self.engine.is_some(),
            self.paused,
            self.last_activity.elapsed(),
        )
    }

    fn source(&self) -> Option<&SourceDescription> {
        self.engine.as_ref().map(|e| &e.source)
    }

    fn duration(&self) -> ClockTime {
        self.source().map(|s| s.duration).unwrap_or(ClockTime::NONE)
    }

    /// The open file's length in nanoseconds, or zero when it never said.
    fn duration_ns(&self) -> u64 {
        self.duration().to_option().map(|d| d.nanos()).unwrap_or(0)
    }

    /// Decode one seek-bar preview, off the interface's thread (#282).
    ///
    /// The first call for a file also opens the second demuxer and decoder,
    /// which is why this is on `spawn_blocking` and not merely "fast enough":
    /// an MP4 index parse is 100 ms and more.
    fn preview(&mut self, request: thumbnail::Request) -> Task<Message> {
        let thumbnail::Request::Decode { at_ns, generation } = request else {
            return Task::none();
        };
        let Some(path) = self.previews.path().map(PathBuf::from) else {
            return Task::none();
        };
        let slot = self.previews.decoder_slot();

        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || thumbnail::decode(&slot, &path, at_ns))
                    .await
                    .unwrap_or_else(|e| (true, Err(format!("the preview task panicked: {e}"))))
            },
            move |(opened, result)| Message::Previewed(generation, at_ns, opened, result),
        )
    }

    fn mailbox(&self) -> Option<Arc<FrameMailbox>> {
        self.engine.as_ref().map(|e| Arc::clone(&e.mailbox))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Enqueue(paths) => {
                self.activity();
                let first = self.playlist.extend(paths);
                // Adding to an idle player starts it; adding to a playing one
                // queues, which is what a drop onto a running film should do.
                match (first, self.playlist.current()) {
                    (Some(index), None) => self.play(index),
                    _ => Task::none(),
                }
            }
            Message::PlayIndex(index) => {
                self.activity();
                self.failures = 0;
                self.play(index)
            }
            Message::Remove(index) => {
                self.activity();
                if self.playlist.remove(index) {
                    // What was playing is gone; stop rather than jump to
                    // whatever slid into its index.
                    self.stop_playback()
                } else {
                    Task::none()
                }
            }
            Message::Next => {
                self.activity();
                self.failures = 0;
                match self.playlist.next() {
                    Some(index) => self.play(index),
                    None => Task::none(),
                }
            }
            Message::Previous => {
                self.activity();
                self.failures = 0;
                match self.playlist.previous() {
                    Some(index) => self.play(index),
                    None => Task::none(),
                }
            }
            Message::CycleRepeat => {
                self.activity();
                self.playlist.repeat = self.playlist.repeat.next();
                if self.options.gapless_loop {
                    // `--repeat` loops inside the demuxer, which never emits
                    // end-of-stream — so nothing here will ever be consulted.
                    // Saying so beats a control that silently does nothing.
                    self.note(
                        Level::Warning,
                        "--repeat loops inside the demuxer, so the playlist never advances",
                    );
                }
                Task::none()
            }
            Message::TogglePlaylist => {
                self.activity();
                self.show_playlist = !self.show_playlist;
                Task::none()
            }
            Message::SelectTrack(kind, id) => {
                self.activity();
                let slot = match kind {
                    engine::TrackKind::Video => &mut self.options.video_track,
                    engine::TrackKind::Audio => &mut self.options.audio_track,
                    // #279 wires the subtitle pad; until then there is
                    // nothing to select one *into*.
                    engine::TrackKind::Subtitle => &mut self.options.subtitle_track,
                };
                // Choosing the track already playing turns it off, which is
                // how "no subtitles" is reachable without a separate control.
                *slot = if *slot == Some(id) { None } else { Some(id) };
                self.reopen_current()
            }
            Message::AudioDevice(device) => {
                self.activity();
                if self.options.audio_device == device {
                    return Task::none();
                }
                self.options.audio_device = device;
                // The graph has no dynamic re-link, and an ALSA sink is
                // opened once for a fixed device — so this is a rebuild, and
                // it resumes where it was rather than starting the file over.
                self.reopen_current()
            }
            Message::TornDown => self.torn_down(),
            Message::Probed(seq, result) => self.build(seq, result),
            Message::Tick => {
                let task = self.poll();
                if self.deadline.is_some_and(|at| Instant::now() >= at) {
                    self.report();
                    return iced::exit();
                }
                task
            }
            // The subscription's only job is to wake an interface that had
            // nothing queued; the widget reads the mailbox itself.
            Message::Frame => Task::none(),
            Message::TogglePause => {
                self.activity();
                if let Some(engine) = &self.engine {
                    if self.paused {
                        engine.handle().resume();
                    } else {
                        engine.handle().pause();
                    }
                    self.paused = !self.paused;
                }
                Task::none()
            }
            Message::SeekBy(secs) => {
                self.activity();
                self.seek_relative(secs);
                Task::none()
            }
            Message::StepBy(frames) => {
                self.activity();
                self.step(frames)
            }
            Message::SeekTo(fraction) => {
                self.activity();
                if let Some(total) = self.duration().to_option() {
                    let target = (fraction.clamp(0.0, 1.0) as f64 * total.nanos() as f64) as u64;
                    self.aim(target);
                }
                Task::none()
            }
            Message::VolumeTo(db) => {
                self.activity();
                self.volume_db = db.clamp(-40.0, 6.0);
                self.muted = false;
                self.apply_volume();
                Task::none()
            }
            Message::Activity => {
                self.activity();
                Task::none()
            }
            Message::VolumeBy(delta) => {
                self.activity();
                self.volume_db = (self.volume_db + delta).clamp(-40.0, 6.0);
                // Reaching for the volume always unmutes: the user wants to
                // hear it.
                self.muted = false;
                self.apply_volume();
                Task::none()
            }
            Message::ToggleMute => {
                self.activity();
                self.muted = !self.muted;
                self.apply_volume();
                Task::none()
            }
            Message::ToggleFullscreen => {
                self.activity();
                self.fullscreen = !self.fullscreen;
                let mode = if self.fullscreen {
                    iced::window::Mode::Fullscreen
                } else {
                    iced::window::Mode::Windowed
                };
                iced::window::latest().and_then(move |id| iced::window::set_mode(id, mode))
            }
            Message::CycleSpeed => {
                self.activity();
                let next = SPEEDS
                    .iter()
                    .copied()
                    .find(|s| *s > self.speed + 1e-6)
                    .unwrap_or(SPEEDS[0]);
                self.set_speed(next);
                Task::none()
            }
            Message::SpeedTo(rate) => {
                self.activity();
                self.set_speed(rate);
                Task::none()
            }
            Message::OpenDialog => {
                self.activity();
                Task::perform(dialog::open_files(), |result| match result {
                    Ok(paths) => Message::Enqueue(paths),
                    // The portal is missing or the user cancelled; either way
                    // there is nothing to open and something to say.
                    Err(reason) => Message::DialogFailed(reason),
                })
            }
            Message::DialogFailed(reason) => {
                self.note(Level::Warning, reason);
                Task::none()
            }
            Message::Escape => {
                self.activity();
                if self.fullscreen {
                    return self.update(Message::ToggleFullscreen);
                }
                iced::exit()
            }
            Message::ToggleStats => {
                self.activity();
                self.show_stats = !self.show_stats;
                // Ask once, the first time anyone looks. It is a round trip
                // through the runtime, and a player that never opens the
                // panel should never pay for it.
                if self.show_stats && self.audio_devices.is_empty() {
                    // Enumerating opens no PCM — it walks ALSA's hint list —
                    // but it is still work nobody who never opens the panel
                    // should pay for.
                    self.audio_devices =
                        parallax::elements::AlsaSink::enumerate_devices().unwrap_or_default();
                }
                if self.show_stats && self.diagnostics.graphics.is_none() {
                    return iced::system::information()
                        .map(|info| Message::Graphics(Box::new(info)));
                }
                Task::none()
            }
            Message::Graphics(info) => {
                self.diagnostics.graphics =
                    Some((info.graphics_adapter.clone(), info.graphics_backend.clone()));
                Task::none()
            }
            Message::DumpGraph => {
                self.activity();
                self.dump_graph();
                Task::none()
            }
            Message::SubtitleSize(delta) => {
                self.activity();
                self.subtitle_size = (self.subtitle_size + delta).clamp(12.0, 64.0);
                Task::none()
            }
            Message::ScrubHover(x) => {
                // Counts as activity: the controls auto-hide on a timer, and a
                // bar that vanished from under the cursor the user is aiming
                // with would take the preview with it. A still cursor produces
                // no events, so this cannot hold them open by itself.
                self.activity();
                let request = self.previews.hover(x, self.duration_ns());
                self.preview(request)
            }
            Message::ScrubLeave => {
                self.previews.leave();
                Task::none()
            }
            Message::ScrubberSized(size) => {
                self.previews.set_bar_width(size.width);
                Task::none()
            }
            Message::Previewed(generation, at_ns, opened, result) => {
                if let Err(why) = &result {
                    tracing::debug!("preview at {at_ns} ns failed: {why}");
                }
                let next = self.previews.finished(generation, at_ns, result, opened);
                self.preview(next)
            }
            Message::Snapshot => {
                self.activity();
                self.snapshot()
            }
            Message::Snapshotted(result) => {
                match result {
                    Ok(path) => self.note(Level::Info, format!("saved {}", path.display())),
                    Err(e) => self.note(Level::Warning, e),
                }
                Task::none()
            }
            Message::Dismiss(index) => {
                self.activity();
                if index < self.notices.len() {
                    self.notices.remove(index);
                }
                Task::none()
            }
            Message::Quit => iced::exit(),
        }
    }

    /// Play the entry at `index`.
    ///
    /// The whole open/teardown chain runs through here, and the two rules
    /// that keep it honest are both about what *cannot* happen. An `Engine`
    /// is never dropped without `teardown()` — `PipelineHandle` has no
    /// `Drop`, so dropping one detaches the tasks and leaves a live pipeline
    /// with an open sound device behind. And there is never more than one
    /// teardown in flight: pressing Next three times during one collapses
    /// into a single teardown of the running file and a single probe of the
    /// third target, rather than three overlapping builds.
    ///
    /// Teardown is two steps rather than one because it is bounded-but-slow
    /// (it waits for the tasks to wind down, up to five seconds) and the
    /// interface must stay live through it. The engine is taken *now*, so
    /// `view()` shows the placeholder immediately rather than the last frame
    /// of the previous file.
    fn play(&mut self, index: usize) -> Task<Message> {
        let Some(path) = self.playlist.select(index) else {
            return Task::none();
        };
        self.seq += 1;
        let request = Request {
            seq: self.seq,
            index,
            path,
        };
        self.status = format!("opening {}…", request.path.display());
        self.reset_transport();

        match self.stage.open() {
            Open::Teardown => {
                let engine = self
                    .engine
                    .take()
                    .expect("Stage::Playing means there is an engine");
                self.stage = Stage::TearingDown {
                    next: Some(request),
                };
                Task::future(async move {
                    engine.teardown().await;
                    Message::TornDown
                })
            }
            // A teardown is already winding the previous pipeline down.
            // Replace its target and spawn nothing: exactly one teardown is
            // in flight, so its `TornDown` is unambiguous when it lands.
            Open::Wait => {
                self.stage = Stage::TearingDown {
                    next: Some(request),
                };
                Task::none()
            }
            // Nothing is running; the outstanding probe (if any) answers a
            // seqnum nobody is waiting for now, and is dropped when it lands.
            Open::Probe => {
                debug_assert!(self.engine.is_none(), "no engine outside Stage::Playing");
                self.stage = Stage::Probing(request.clone());
                probe(request)
            }
        }
    }

    /// The previous pipeline is gone; start whatever was waiting for it.
    fn torn_down(&mut self) -> Task<Message> {
        let Stage::TearingDown { next } = std::mem::replace(&mut self.stage, Stage::Idle) else {
            return Task::none();
        };
        match next {
            Some(request) => {
                self.stage = Stage::Probing(request.clone());
                probe(request)
            }
            None => Task::none(),
        }
    }

    /// Build and start the pipeline for a probed file.
    ///
    /// Synchronously, in `update`: `Executor::start` is sync and fast, and
    /// Iced's `tokio` feature runs `update` inside `runtime.enter()`, so the
    /// tasks it spawns find a runtime. Doing it here is what keeps `Engine`
    /// and `PipelineHandle` off the `Send` hook.
    fn build(&mut self, seq: u64, result: ProbeCell) -> Task<Message> {
        // Only the probe the user is actually waiting for gets to build. An
        // older one answers a request that has already been superseded, and
        // building it would put the wrong file on screen — or, worse, over a
        // pipeline that is already running.
        if !self.stage.accepts_probe(seq) {
            return Task::none();
        }
        let Stage::Probing(request) = &self.stage else {
            unreachable!("accepts_probe implies Probing")
        };
        let request = request.clone();
        // A cloned message finds the cell already emptied. Iced clones
        // messages on some paths, and the alternative — `Arc::try_unwrap`,
        // which reports a clone as the error "probe result was cloned" —
        // turns a harmless duplicate into a failed open.
        let Some(result) = result.take() else {
            return Task::none();
        };
        self.stage = Stage::Idle;
        debug_assert!(self.engine.is_none(), "building over a live engine");

        let probed = match result {
            Ok(p) => p,
            Err(e) => {
                self.note(Level::Error, format!("{}: {e}", request.path.display()));
                self.status = "no file open".into();
                return self.skip_failed(request.index);
            }
        };

        match Engine::build(probed, &self.options) {
            Ok(engine) => {
                self.status = engine.source.summary();
                if !engine.has_audio() {
                    self.status.push_str(" · video only");
                }
                self.apply_volume_to(&engine);
                self.engine = Some(engine);
                self.stage = Stage::Playing;
                // Previews are for *this* file now. Nothing is opened here —
                // the second demuxer costs another index parse, and a file
                // nobody scrubs should not pay it (#282).
                self.previews.open(request.path.clone());
                recent::remember(&request.path);
                self.recent = recent::load();
                if let Some(at) = self.resume_at.take() {
                    self.aim(at);
                }
                // The transport's state belongs to the *player*, not to the
                // pipeline that happens to be running: a rate set before a
                // track change was still shown afterwards while the new
                // pipeline played at 1x, because nothing re-applied it.
                if self.speed != 1.0 {
                    let rate = self.speed;
                    self.set_speed(rate);
                }
                Task::none()
            }
            Err(e) => {
                self.note(Level::Error, format!("{e:#}"));
                self.status = "no file open".into();
                // Whatever it was resuming to belongs to a pipeline that
                // never started; the next file starts at its own beginning.
                self.resume_at = None;
                self.skip_failed(request.index)
            }
        }
    }

    /// An entry could not be opened: mark it, and move on if there is a list.
    ///
    /// "Could not be opened" is deliberately not the same as "did not play".
    /// The counter that stops the skipping is reset when a file actually
    /// presents a frame, not when it builds — otherwise a directory of files
    /// that open and die 200 ms later would be walked forever.
    fn skip_failed(&mut self, index: usize) -> Task<Message> {
        self.playlist.mark_failed(index);
        self.failures += 1;
        if give_up(self.failures, self.playlist.len()) {
            self.note(
                Level::Error,
                "nothing in the playlist could be played — stopping",
            );
            return Task::none();
        }
        if self.playlist.len() <= 1 {
            return Task::none();
        }
        match self.playlist.advance() {
            Some(next) => self.play(next),
            None => Task::none(),
        }
    }

    /// Play whatever comes after the file that just ended.
    fn advance(&mut self) -> Task<Message> {
        match self.playlist.advance() {
            Some(next) => self.play(next),
            None => {
                self.status = "end of playlist".into();
                Task::none()
            }
        }
    }

    /// One machine-readable line, for `scripts/decode_matrix.py` and #270.
    fn report(&self) {
        let stats = self
            .engine
            .as_ref()
            .map(|e| e.mailbox.stats())
            .unwrap_or_default();
        let dropped = self.engine.as_ref().map(|e| e.dropped()).unwrap_or(0);
        println!("{}", stats_line(self.position, dropped, &stats));
    }

    /// Save the frame on screen (#283).
    ///
    /// The frame comes from the mailbox, not from a branch in the pipeline:
    /// it is the decoder's own output, so what lands in the file is the
    /// source frame at full resolution rather than the scaled, letterboxed,
    /// overlaid window. Converting and compressing it is real work, so it
    /// goes to a blocking thread — the interface must not stutter for a
    /// keypress.
    fn snapshot(&mut self) -> Task<Message> {
        let Some(engine) = &self.engine else {
            self.note(Level::Warning, "nothing is playing");
            return Task::none();
        };
        let Some(frame) = engine.mailbox.last_presented() else {
            self.note(Level::Warning, "no frame has been shown yet");
            return Task::none();
        };
        let source = engine.source.path.clone();
        let at = self.position.to_option().map(|p| p.nanos()).unwrap_or(0);

        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    snapshot::write_png(&frame, &source, at).map_err(|e| format!("{e:#}"))
                })
                .await
                .unwrap_or_else(|e| Err(format!("the snapshot task panicked: {e}")))
            },
            Message::Snapshotted,
        )
    }

    /// Write the running graph out as GraphViz, and say where it went.
    ///
    /// `PARALLAX_DOT_DIR` is the engine's own convention for state-transition
    /// dumps, so a dump asked for by hand lands in the same place.
    fn dump_graph(&mut self) {
        let Some(engine) = &self.engine else {
            self.note(Level::Warning, "nothing is playing");
            return;
        };
        let dir = std::env::var("PARALLAX_DOT_DIR").unwrap_or_else(|_| ".".into());
        let path = std::path::Path::new(&dir).join(format!("parallax-{}.dot", std::process::id()));
        let (level, text) = match std::fs::write(&path, engine.dot()) {
            Ok(()) => (Level::Info, format!("wrote {}", path.display())),
            Err(e) => (
                Level::Warning,
                format!("could not write {}: {e}", path.display()),
            ),
        };
        self.note(level, text);
    }

    fn reset_transport(&mut self) {
        self.position = ClockTime::ZERO;
        self.paused = false;
        self.seek = SeekState::default();
        // Dropping the thumbnailer here is what "closing the player leaves no
        // orphaned decoder" means (#282): it owns a `File`, a demuxer index
        // and a decoder with its arena, none of which the next file wants.
        self.previews.close();
    }

    /// Everything the interface re-reads on a tick.
    ///
    /// Returns whatever the pipeline's news asks for — today, opening the
    /// next file when this one ends.
    fn poll(&mut self) -> Task<Message> {
        self.expire_notices();

        let Some(engine) = &mut self.engine else {
            return Task::none();
        };
        self.position = engine.position();
        // A file that reached the screen is a file that played, whatever
        // happens to it next: that is what stops the failure-skipping.
        if engine.mailbox.stats().presented > 0 {
            self.failures = 0;
        }

        // The cues on screen, against the same running time the widget paces
        // to — so a subtitle and the frame it belongs to appear together.
        let now = engine
            .mailbox
            .clock()
            .map(|clock| clock.now())
            .unwrap_or(ClockTime::NONE);
        self.cues = engine.cues.active(now);

        let mut ended = false;
        let messages = engine.poll_bus();
        for message in &messages {
            match &message.kind {
                MessageKind::SeekDone { seqnum, .. } => {
                    if self.seek.in_flight.map(|(s, _)| s) == Some(*seqnum) {
                        self.seek.in_flight = None;
                        self.seek.sent = None;
                    }
                }
                MessageKind::Error { error, .. } => {
                    self.note(Level::Error, format!("{}: {error}", message.source));
                }
                MessageKind::Warning { warning, .. } => {
                    tracing::warn!("{warning}");
                    self.note(Level::Warning, format!("{}: {warning}", message.source));
                }
                MessageKind::Info { info } => self.note(Level::Info, info.clone()),
                MessageKind::Buffering { percent, .. } => {
                    self.diagnostics.buffering = Some(*percent);
                }
                MessageKind::DownloadProgress { ranges, total, .. } => {
                    let have = ranges.iter().map(|(from, to)| to - from).sum();
                    self.diagnostics.downloaded = Some((have, *total));
                }
                MessageKind::Qos {
                    proportion,
                    jitter_ns,
                    ..
                } => {
                    // A rate advisory, not an event: the newest one is the
                    // only one worth showing.
                    self.diagnostics.qos = Some((*proportion, *jitter_ns));
                }
                MessageKind::NewClock { clock_name } => {
                    self.diagnostics.clock = Some(clock_name.clone());
                }
                MessageKind::ClockLost => self.diagnostics.clock = None,
                MessageKind::DurationChanged {
                    duration: Some(duration),
                } => {
                    // A duration the container did not know at probe time.
                    // Without this the seek bar stays a "live / unknown
                    // length" label for a file that has since said.
                    if let Some(engine) = &mut self.engine {
                        engine.source.duration = *duration;
                    }
                }
                // The end of the stream, as an *edge*. Deliberately not
                // `mailbox.end_reason()`, which is a level that stays set —
                // and which the sink raises when the EOS *event* arrives,
                // with frames still queued and the whole audio device buffer
                // still to play. This is posted after every task finished.
                //
                // It is also scoped to this engine: the bus dies with it, so
                // the `Eos` that `stop()` posts while tearing a file down
                // cannot reach the player and advance the playlist a second
                // time. A `PipelineHandle::ended()` future would.
                MessageKind::Eos => ended = true,
                _ => {}
            }
        }

        if let Some(engine) = &self.engine
            && let Some(reason) = engine.mailbox.end_reason()
        {
            self.status = format!("{} · {reason:?}", engine.source.summary());
        }

        self.dispatch_seek();

        if ended && matches!(self.stage, Stage::Playing) {
            return self.advance();
        }
        Task::none()
    }

    /// Rebuild the current file, resuming where it was.
    ///
    /// The whole graph is fixed at `start()`: no dynamic re-link, and the
    /// ALSA sink is opened once for one device. Anything that changes how the
    /// pipeline is *built* therefore goes through here, which is a visible
    /// hitch and honest about being one.
    fn reopen_current(&mut self) -> Task<Message> {
        let Some(index) = self.playlist.current() else {
            return Task::none();
        };
        self.resume_at = self
            .seek
            .target()
            .or_else(|| self.position.to_option().map(|p| p.nanos()))
            .filter(|at| *at > 0);
        self.play(index)
    }

    /// Stop playing, without opening anything else.
    fn stop_playback(&mut self) -> Task<Message> {
        self.status = "no file open".into();
        self.reset_transport();
        match std::mem::replace(&mut self.stage, Stage::Idle) {
            Stage::Playing => {
                let engine = self
                    .engine
                    .take()
                    .expect("Stage::Playing means there is an engine");
                self.stage = Stage::TearingDown { next: None };
                Task::future(async move {
                    engine.teardown().await;
                    Message::TornDown
                })
            }
            Stage::TearingDown { .. } => {
                self.stage = Stage::TearingDown { next: None };
                Task::none()
            }
            other => {
                self.stage = other;
                Task::none()
            }
        }
    }

    /// Say something, keeping the last few.
    fn note(&mut self, level: Level, text: impl Into<String>) {
        let text = text.into();
        let at = Instant::now();
        // A stream that repeats itself should not fill the screen with the
        // same sentence.
        if self.notices.back().is_some_and(|last| last.text == text) {
            if let Some(last) = self.notices.back_mut() {
                last.at = at;
            }
            return;
        }
        self.notices.push_back(Notice { text, level, at });
        while self.notices.len() > NOTICE_LIMIT {
            self.notices.pop_front();
        }
    }

    fn expire_notices(&mut self) {
        let now = Instant::now();
        self.notices.retain(|notice| {
            !notice_expired(notice.level, now.saturating_duration_since(notice.at))
        });
    }

    fn seek_relative(&mut self, secs: i64) {
        // From where the last press aimed, not from `position()`: until the
        // seek's Segment lands that still reports the old position, so a
        // second press would walk from there and undo the first.
        let from = self.seek.target().map(|t| t as i64).unwrap_or_else(|| {
            self.position
                .to_option()
                .map(|p| p.nanos() as i64)
                .unwrap_or(0)
        });
        let mut target = (from + secs * 1_000_000_000).max(0);
        // A duration of NONE means *unknown* (MKV with no Segment Duration),
        // not an empty file — clamping against zero sends every seek to the
        // start.
        if let Some(total) = self.duration().to_option() {
            target = target.min(total.nanos().saturating_sub(1) as i64);
        }
        self.aim(target as u64);
    }

    fn aim(&mut self, target_ns: u64) {
        self.seek.pending = Some(target_ns);
        self.seek.due = Some(Instant::now() + SEEK_DEBOUNCE);
        self.seek.accurate = false;
    }

    /// Step one frame, pausing first (#281).
    ///
    /// A flushing `ACCURATE` seek per frame is wasteful — a flush trio, a
    /// decoder reset and a keyframe re-decode to advance by one frame
    /// duration — but it is correct, and it is built out of a path that
    /// already works. The cheaper answer is a real `Event::Step` that a paced
    /// sink consumes by releasing exactly N frames; that design is written up
    /// in `plans/30_iced_player.md` and deliberately not built here.
    ///
    /// Backward stepping is a backward seek like any other. Decoding
    /// backwards is what the engine cannot do; seeking backwards it does
    /// every time an arrow key is pressed.
    fn step(&mut self, frames: i32) -> Task<Message> {
        let Some(frame) = self
            .source()
            .and_then(|s| s.video.as_ref())
            .and_then(|v| v.frame_duration())
        else {
            // Guessing 30 fps here would step 33 ms and call it a frame. The
            // container said nothing; say so.
            self.note(
                Level::Warning,
                "this file does not state a frame rate, so it cannot be stepped",
            );
            return Task::none();
        };

        // Stepping is a paused gesture: it is how you look at one frame.
        let task = if self.paused {
            Task::none()
        } else {
            self.update(Message::TogglePause)
        };

        let target = step_target(self.step_origin(), frame, frames, self.duration());

        // No debounce: holding `.` should step frame by frame, not coalesce
        // four presses into one. And ACCURATE, or the seek lands on the
        // previous keyframe and the picture goes backwards instead of on.
        self.seek.pending = Some(target);
        self.seek.due = None;
        self.seek.accurate = true;
        self.dispatch_seek();
        task
    }

    /// Where a step counts from.
    ///
    /// The last aim if there is one, for the same reason `seek_relative`
    /// walks from it: `position()` does not move until the seek's Segment
    /// lands, so counting from it would make the second of two quick steps
    /// undo the first.
    fn step_origin(&self) -> i64 {
        self.seek.target().map(|t| t as i64).unwrap_or_else(|| {
            self.position
                .to_option()
                .map(|p| p.nanos() as i64)
                .unwrap_or(0)
        })
    }

    /// Send the accumulated seek, once the debounce has expired and the last
    /// one is out of the way.
    fn dispatch_seek(&mut self) {
        let Some(engine) = &self.engine else {
            return;
        };
        let now = Instant::now();
        if self.seek.due.is_some_and(|due| now < due) {
            return;
        }
        let Some(target) = self.seek.pending else {
            return;
        };
        // At most one in flight — but do not wait forever for a `SeekDone`
        // that may have been lost. Dispatching twice is harmless: the engine
        // takes the newest.
        if self.seek.in_flight.is_some()
            && self.seek.sent.is_some_and(|at| now - at < SEEK_SUPERSEDE)
        {
            return;
        }

        let mut event = SeekEvent::new_time(ClockTime::from_nanos(target));
        if self.seek.accurate {
            // `new_time` is FLUSH | KEY_UNIT, and `with_flags` replaces
            // rather than ORs — so the whole set has to be restated.
            //
            // SNAP_BEFORE matters as much as ACCURATE here, and it is not
            // obvious: the per-container default snap is *forward* for MKV,
            // so a step from 600 ms to 700 ms would land on the keyframe at
            // 1000 ms — a step forward that skips most of a GOP. Snapping
            // backward puts the decoder at the keyframe before the target and
            // ACCURATE makes it decode-but-drop the gap, so the frame that
            // appears is the one that was asked for.
            event = event.with_flags(
                SeekFlags::FLUSH
                    | SeekFlags::KEY_UNIT
                    | SeekFlags::ACCURATE
                    | SeekFlags::SNAP_BEFORE,
            );
        }
        let seqnum = event.seqnum();
        if engine.handle().dispatch_seek(event) {
            self.seek.in_flight = Some((seqnum, target));
            self.seek.sent = Some(now);
        } else {
            self.note(Level::Warning, "this source cannot seek");
        }
        self.seek.pending = None;
        self.seek.due = None;
        self.seek.accurate = false;
    }

    /// Ask the pipeline to play at `rate` (#280).
    ///
    /// An `INSTANT_RATE_CHANGE` seek: no flush, no repositioning, and it
    /// terminates at the sinks — so it works on a live pipeline that cannot
    /// seek at all. Audio goes silent above or below 1x, because `AlsaSink`
    /// plays at the hardware rate and there is no pitch-preserving
    /// time-stretch in the tree; it keeps writing silence at the running-time
    /// rate so the clock — which that sink provides — stays true and video
    /// stays locked to it.
    fn set_speed(&mut self, rate: f64) {
        let Some(engine) = &self.engine else {
            return;
        };
        if engine
            .handle()
            .dispatch_seek(SeekEvent::new_instant_rate(rate))
        {
            self.speed = rate;
        } else {
            // A sign flip, or nothing playing yet. Reverse needs a real seek.
            self.note(Level::Warning, format!("cannot play at {rate}x"));
        }
    }

    fn apply_volume(&self) {
        if let Some(engine) = &self.engine {
            self.apply_volume_to(engine);
        }
    }

    fn apply_volume_to(&self, engine: &Engine) {
        let factor = if self.muted {
            0.0
        } else {
            10f32.powf(self.volume_db / 20.0)
        };
        engine.set_volume(factor);
    }

    fn view(&self) -> Element<'_, Message> {
        let video: Element<'_, Message> = match self.mailbox() {
            Some(mailbox) => iced::widget::shader(parallax_iced::VideoView::new(mailbox))
                .width(iced::Fill)
                .height(iced::Fill)
                .into(),
            None => center(ui::welcome(self)).into(),
        };

        let video = iced::widget::mouse_area(video)
            .on_move(|_| Message::Activity)
            .on_press(Message::TogglePause)
            .on_double_click(Message::ToggleFullscreen)
            .on_scroll(|delta| Message::VolumeBy(volume_step(delta)))
            // The pointer goes away with the chrome, under the same rule:
            // both are things that are not the film. `MouseArea` only
            // overrides an interaction the content reported as `None`, which
            // is what `shader` reports, so the video surface is the only
            // place this applies — a button still shows a pointer.
            .interaction(if self.controls_visible() {
                iced::mouse::Interaction::None
            } else {
                iced::mouse::Interaction::Hidden
            });

        let mut layers = stack![
            container(video)
                .width(iced::Fill)
                .height(iced::Fill)
                .style(|_| container::background(iced::Color::BLACK)),
        ];
        // Subtitles sit above the picture and *below* the transport, so a cue
        // can never occlude the controls (#279). Mounted only when a cue is
        // actually on screen: `ui.rs`'s rule is that the overlay is unmounted
        // rather than transparent, because every redraw runs full layout and
        // text shaping at the frame rate.
        if !self.cues.is_empty() {
            layers = layers.push(ui::subtitles(self));
        }
        layers.push(ui::overlay(self)).into()
    }

    fn subscription(&self) -> Subscription<Message> {
        let mut subs = vec![
            iced::time::every(TICK).map(|_| Message::Tick),
            keyboard::listen()
                .with(())
                .filter_map(|(_, event)| match event {
                    keyboard::Event::KeyPressed { key, .. } => shortcut(&key),
                    _ => None,
                }),
        ];
        // Only while there is a stream: the recipe is keyed on the mailbox's
        // id, so a rebuild swaps it and frames from the old file cannot reach
        // the new interface.
        // Files dropped onto the window. Iced emits one event per file, so a
        // multi-file drop enqueues naturally.
        //
        // Wayland does not deliver these at all — `window::Event::FileDropped`
        // says so in its own documentation, and this player enables both
        // backends. It works on X11; the file dialog and the playlist are
        // what actually have to carry opening a file.
        subs.push(iced::window::events().filter_map(|(_, event)| match event {
            iced::window::Event::FileDropped(path) => Some(Message::Enqueue(vec![path])),
            _ => None,
        }));

        if let Some(mailbox) = self.mailbox() {
            subs.push(parallax_iced::frames(mailbox).map(|_| Message::Frame));
        }
        Subscription::batch(subs)
    }
}

/// The key map. Deliberately the one the old player had, plus the diagnostics
/// panel — muscle memory is a feature.
fn shortcut(key: &Key) -> Option<Message> {
    match key {
        Key::Named(Named::Space) => Some(Message::TogglePause),
        Key::Named(Named::ArrowLeft) => Some(Message::SeekBy(-10)),
        Key::Named(Named::ArrowRight) => Some(Message::SeekBy(10)),
        Key::Named(Named::ArrowUp) => Some(Message::VolumeBy(2.0)),
        Key::Named(Named::ArrowDown) => Some(Message::VolumeBy(-2.0)),
        // Escape *leaves* fullscreen rather than toggling into it, which is
        // what every other application does with it; `f` is the toggle. What
        // it does from a window is decided in `update`, where the state is.
        Key::Named(Named::Escape) => Some(Message::Escape),
        Key::Character(c) => match c.as_str() {
            "q" => Some(Message::Quit),
            "f" => Some(Message::ToggleFullscreen),
            "m" => Some(Message::ToggleMute),
            "i" => Some(Message::ToggleStats),
            "." => Some(Message::StepBy(1)),
            "," => Some(Message::StepBy(-1)),
            "c" => Some(Message::Snapshot),
            "d" => Some(Message::DumpGraph),
            "+" | "=" => Some(Message::SubtitleSize(2.0)),
            "-" => Some(Message::SubtitleSize(-2.0)),
            "o" => Some(Message::OpenDialog),
            "p" => Some(Message::TogglePlaylist),
            "n" => Some(Message::Next),
            "b" => Some(Message::Previous),
            "r" => Some(Message::CycleRepeat),
            "s" => Some(Message::CycleSpeed),
            " " => Some(Message::TogglePause),
            _ => None,
        },
        _ => None,
    }
}

/// Whether the transport should be on screen.
///
/// Always, unless something is actually playing: there is nothing to get out
/// of the way of otherwise, and a paused player is a player someone is about
/// to touch.
fn controls_visible(playing: bool, paused: bool, idle: Duration) -> bool {
    !playing || paused || idle < CONTROLS_LINGER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_controls_only_get_out_of_the_way_of_playback() {
        let long = CONTROLS_LINGER + Duration::from_secs(1);
        assert!(
            controls_visible(false, false, long),
            "nothing playing: nothing to hide for"
        );
        assert!(
            controls_visible(true, true, long),
            "paused: someone is about to touch it"
        );
        assert!(controls_visible(true, false, Duration::ZERO), "just used");
        assert!(
            !controls_visible(true, false, long),
            "playing and idle: get out of the way"
        );
    }

    #[test]
    fn an_unknown_duration_reads_as_unknown_not_as_zero() {
        // The distinction matters beyond cosmetics: clamping a seek against a
        // zero duration sends every one of them to the start.
        assert_eq!(ui::timecode(ClockTime::NONE), "--:--");
        assert_eq!(ui::timecode(ClockTime::ZERO), "00:00");
        assert_eq!(ui::timecode(ClockTime::from_millis(125_500)), "02:05");
    }

    #[test]
    fn a_seek_target_leads_the_position_it_has_not_reached_yet() {
        let mut seek = SeekState::default();
        assert_eq!(seek.target(), None);
        seek.in_flight = Some((7, 5_000_000_000));
        assert_eq!(seek.target(), Some(5_000_000_000));
        // A newer press supersedes the one in flight, so the bar leads to the
        // newest target rather than snapping back to the old one.
        seek.pending = Some(9_000_000_000);
        assert_eq!(seek.target(), Some(9_000_000_000));
    }

    #[test]
    fn the_key_map_is_the_one_people_already_know() {
        use iced::keyboard::key::Named;
        assert!(matches!(
            shortcut(&Key::Named(Named::Space)),
            Some(Message::TogglePause)
        ));
        assert!(matches!(
            shortcut(&Key::Named(Named::ArrowLeft)),
            Some(Message::SeekBy(-10))
        ));
        assert!(matches!(
            shortcut(&Key::Character("q".into())),
            Some(Message::Quit)
        ));
        assert!(matches!(
            shortcut(&Key::Character("s".into())),
            Some(Message::CycleSpeed)
        ));
        assert!(matches!(
            shortcut(&Key::Character("n".into())),
            Some(Message::Next)
        ));
        assert!(matches!(
            shortcut(&Key::Character("b".into())),
            Some(Message::Previous)
        ));
        assert!(matches!(
            shortcut(&Key::Character("p".into())),
            Some(Message::TogglePlaylist)
        ));
        assert!(matches!(
            shortcut(&Key::Character(".".into())),
            Some(Message::StepBy(1))
        ));
        assert!(matches!(
            shortcut(&Key::Character(",".into())),
            Some(Message::StepBy(-1))
        ));
        assert!(shortcut(&Key::Character("z".into())).is_none());
    }

    /// `s` walks the speeds and wraps, so one key reaches every rate (#280).
    #[test]
    fn the_speed_key_cycles_through_every_rate_and_wraps() {
        // The table is deliberately not sorted: 1x first, because that is
        // where a player starts and where a user most wants to get back to.
        // The cycle follows the table's own order.
        let next = |from: f64| {
            SPEEDS
                .iter()
                .copied()
                .find(|s| *s > from + 1e-6)
                .unwrap_or(SPEEDS[0])
        };
        assert_eq!(next(1.0), 1.25);
        assert_eq!(next(1.25), 1.5);
        assert_eq!(next(1.5), 2.0);
        // Past the fastest it wraps to the front rather than sticking.
        assert_eq!(next(2.0), 1.0);
        // Half speed is below everything, so it steps up into the sequence.
        assert_eq!(next(0.5), 1.0);
        assert!(SPEEDS.iter().all(|s| *s > 0.0), "no reverse: see SPEEDS");
    }

    /// The rule that keeps two pipelines from being alive at once (#276).
    ///
    /// `PipelineHandle` has no `Drop`: dropping one detaches its tasks and
    /// leaves a live pipeline behind, sound device and all. So an engine
    /// exists only in `Stage::Playing`, and every path out of it goes through
    /// a teardown — of which there is never more than one.
    #[test]
    fn pressing_next_during_a_teardown_does_not_start_a_second_one() {
        assert_eq!(Stage::Playing.open(), Open::Teardown);
        // The second, third and tenth press all land here: the target is
        // replaced, nothing is spawned.
        assert_eq!(Stage::TearingDown { next: None }.open(), Open::Wait);
        assert_eq!(
            Stage::TearingDown {
                next: Some(request(7))
            }
            .open(),
            Open::Wait
        );
        // Nothing is running, so there is nothing to wind down first.
        assert_eq!(Stage::Idle.open(), Open::Probe);
        assert_eq!(Stage::Probing(request(1)).open(), Open::Probe);
    }

    /// An Iced task cannot be cancelled, so a superseded probe still answers.
    #[test]
    fn only_the_probe_being_waited_for_is_built() {
        let stage = Stage::Probing(request(4));
        assert!(stage.accepts_probe(4));
        assert!(!stage.accepts_probe(3), "an older request was superseded");
        assert!(!stage.accepts_probe(5), "and there is no newer one yet");

        // Nothing else accepts one at all — building while a pipeline is
        // running or winding down is the leak this prevents.
        assert!(!Stage::Playing.accepts_probe(4));
        assert!(!Stage::Idle.accepts_probe(4));
        assert!(!Stage::TearingDown { next: None }.accepts_probe(4));
    }

    /// A cloned message must be harmless, not an error (#276).
    #[test]
    fn a_probe_result_can_only_be_taken_once() {
        let cell = ProbeCell::new(Err("no".into()));
        let clone = cell.clone();
        assert!(cell.take().is_some());
        // The clone finds it emptied and does nothing, which is right: the
        // request it answers has already been dealt with.
        assert!(clone.take().is_none());
        assert!(cell.take().is_none());
    }

    /// One bad file in a directory is normal; a whole directory of them is
    /// not a reason to spin forever (#276).
    #[test]
    fn skipping_stops_after_a_full_lap_of_failures() {
        // A single file has nowhere to skip to.
        assert!(!give_up(1, 1));
        assert!(!give_up(9, 1));
        // Part way through a list, keep going.
        assert!(!give_up(1, 3));
        assert!(!give_up(2, 3));
        // A full lap with nothing playable is the end of it.
        assert!(give_up(3, 3));
        assert!(give_up(4, 3));
    }

    fn request(seq: u64) -> Request {
        Request {
            seq,
            index: 0,
            path: PathBuf::from("f.mp4"),
        }
    }

    /// Stepping refuses rather than guessing (#281).
    ///
    /// `Framerate::default()` is 30 fps, so "unknown" and "30" are the same
    /// value on the buffer path — which is why the step size comes from the
    /// container header and why `None` has to mean no.
    #[test]
    fn a_file_that_states_no_frame_rate_cannot_be_stepped() {
        let stream = |rate| engine::VideoStream {
            codec: engine::VideoCodecKind::H264,
            width: 64,
            height: 64,
            frame_rate: rate,
        };
        assert_eq!(stream(None).frame_duration(), None);
        assert_eq!(
            stream(Some(25.0)).frame_duration(),
            Some(Duration::from_millis(40))
        );
        // A container stating nonsense has said nothing useful.
        assert_eq!(stream(Some(0.0)).frame_duration(), None);
        assert_eq!(stream(Some(-24.0)).frame_duration(), None);
        assert_eq!(stream(Some(f64::NAN)).frame_duration(), None);
        assert_eq!(stream(Some(f64::INFINITY)).frame_duration(), None);
        // Past any real file, and the guard that keeps steps off zero.
        assert_eq!(stream(Some(100_000.0)).frame_duration(), None);
        assert!(stream(Some(1000.0)).frame_duration().is_some());
    }

    /// Where a step lands (#281).
    #[test]
    fn a_step_moves_by_exactly_one_frame_and_stops_at_the_ends() {
        let frame = Duration::from_millis(100);
        let two_seconds = ClockTime::from_nanos(2_000_000_000);

        // Forward and back, one frame at a time.
        assert_eq!(step_target(500_000_000, frame, 1, two_seconds), 600_000_000);
        assert_eq!(
            step_target(500_000_000, frame, -1, two_seconds),
            400_000_000
        );
        assert_eq!(step_target(500_000_000, frame, 4, two_seconds), 900_000_000);

        // The start is a floor, not a wrap.
        assert_eq!(step_target(50_000_000, frame, -1, two_seconds), 0);
        assert_eq!(step_target(0, frame, -10, two_seconds), 0);

        // The end is a ceiling, one nanosecond inside the duration.
        assert_eq!(
            step_target(1_950_000_000, frame, 1, two_seconds),
            1_999_999_999
        );

        // An unknown duration does not clamp to zero — that is the trap a
        // Segment-Duration-less MKV sets, and it would send every step to
        // the start of the file.
        assert_eq!(
            step_target(1_950_000_000, frame, 1, ClockTime::NONE),
            2_050_000_000
        );
    }

    /// A wheel notch is an arrow key, and a touchpad flick is not a lurch.
    #[test]
    fn the_wheel_moves_the_volume_the_way_the_arrow_keys_do() {
        use iced::mouse::ScrollDelta;
        assert_eq!(volume_step(ScrollDelta::Lines { x: 0.0, y: 1.0 }), 2.0);
        assert_eq!(volume_step(ScrollDelta::Lines { x: 0.0, y: -1.0 }), -2.0);
        // Horizontal scrolling is not a volume gesture.
        assert_eq!(volume_step(ScrollDelta::Lines { x: 5.0, y: 0.0 }), 0.0);
        // Pixels: 50 to a notch.
        assert_eq!(volume_step(ScrollDelta::Pixels { x: 0.0, y: 50.0 }), 2.0);
        assert_eq!(volume_step(ScrollDelta::Pixels { x: 0.0, y: 25.0 }), 1.0);
        // An inertial flick is clamped rather than swinging the whole range.
        assert_eq!(volume_step(ScrollDelta::Pixels { x: 0.0, y: 5000.0 }), 6.0);
        assert_eq!(volume_step(ScrollDelta::Lines { x: 0.0, y: -40.0 }), -6.0);
    }

    /// Escape leaves fullscreen; it does not enter it (#274).
    #[test]
    fn escape_is_a_way_out_not_a_toggle() {
        assert!(matches!(
            shortcut(&Key::Named(Named::Escape)),
            Some(Message::Escape)
        ));
        // `f` remains the toggle, in both directions.
        assert!(matches!(
            shortcut(&Key::Character("f".into())),
            Some(Message::ToggleFullscreen)
        ));
        assert!(matches!(
            shortcut(&Key::Character("o".into())),
            Some(Message::OpenDialog)
        ));
    }

    /// A warning is passing news; an error is the end of the run (#275).
    #[test]
    fn an_error_waits_to_be_read_and_a_warning_does_not() {
        // Fresh: everything stays.
        assert!(!notice_expired(Level::Info, Duration::ZERO));
        assert!(!notice_expired(Level::Warning, Duration::from_secs(1)));
        assert!(!notice_expired(Level::Error, Duration::ZERO));

        // Past the linger, only the passing kinds go.
        assert!(notice_expired(Level::Info, NOTICE_LINGER));
        assert!(notice_expired(Level::Warning, NOTICE_LINGER * 2));
        assert!(
            !notice_expired(Level::Error, NOTICE_LINGER * 1000),
            "an error that vanished before it was read is the same as no error"
        );
    }

    /// The `stats:` line is a contract with `scripts/decode_matrix.py`, so it
    /// is pinned here rather than only described in a comment above the
    /// format string.
    #[test]
    fn the_summary_line_is_the_shape_the_matrix_runner_parses() {
        let stats = parallax::video::FrameStats {
            queued: 2,
            presented: 20,
            dropped_late: 3,
            dropped_full: 4,
            dropped_stale: 5,
            clipped: 6,
            worst_late_ns: 7,
        };
        let line = stats_line(ClockTime::from_nanos(1_900_000_000), 1, &stats);
        assert_eq!(
            line,
            "stats: position_ns=1900000000 dropped=1 presented=20 queued=2 \
             late=3 full=4 stale=5 clipped=6"
                .replace("             ", "")
        );
        // The pair the runner's regex requires, adjacent and in this order.
        assert!(line.contains("stats: position_ns=1900000000 dropped=1"));
        // An unknown position reads as zero, not as a panic or a sentinel.
        assert!(
            stats_line(ClockTime::NONE, 0, &Default::default())
                .starts_with("stats: position_ns=0 dropped=0")
        );
    }

    /// The measurement flags the crossterm player had (#277). They vanished
    /// once already, silently, which is why they are asserted rather than
    /// trusted.
    #[test]
    fn the_measurement_flags_survive_the_rewrite() {
        use clap::CommandFactory;
        Args::command().debug_assert();

        let args = Args::try_parse_from([
            "parallax-player",
            "--no-sync",
            "--channel-capacity=16",
            "--frame-delay=2",
            "--headless",
            "movie.mp4",
        ])
        .expect("the measurement flags parse");
        let options = args.options();
        assert!(!options.sync, "--no-sync turns pacing off");
        assert_eq!(options.channel_capacity, 16);
        assert_eq!(options.frame_delay, 2);
        assert!(args.headless);

        // A capacity of zero would be a link that can hold nothing; the
        // executor wants at least one slot.
        let zero = Args::try_parse_from(["parallax-player", "--channel-capacity=0", "f.mp4"])
            .unwrap()
            .options();
        assert_eq!(zero.channel_capacity, 1);

        // `--loop` is what the old player called `--repeat`, and scripts
        // outlive rewrites.
        assert!(
            Args::try_parse_from(["parallax-player", "--loop", "movie.mp4"])
                .expect("--loop still parses")
                .repeat
        );
    }
}
