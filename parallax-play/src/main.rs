//! A video player built on parallax and Iced.
//!
//! The engine does the hard parts — demux, decode, A/V sync, seeking — and
//! this is the front end: a window, a video surface, and the transport.
//!
//! ```text
//! demux ─┬─ video ─▶ decode ─▶ AppVideoSink ⇄ mailbox ⇄ VideoView (this)
//!        └─ audio ─▶ decode ─▶ downmix ─▶ gain ─▶ AlsaSink (the clock)
//! ```

mod engine;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use iced::keyboard::{self, Key, key::Named};
use iced::widget::{center, container, stack, text};
use iced::{Element, Subscription, Task, Theme};

use parallax::clock::ClockTime;
use parallax::event::SeekEvent;
use parallax::pipeline::MessageKind;
use parallax::video::FrameMailbox;

use engine::{Engine, HwDec, Options, Probed, SourceDescription};

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

#[derive(Parser, Debug, Clone)]
#[command(name = "parallax-play", about = "A video player built on parallax")]
struct Args {
    /// MP4/MOV or MKV/WebM file to play.
    file: Option<PathBuf>,

    /// Play without audio.
    #[arg(long)]
    no_audio: bool,

    /// Loop the file, gaplessly, inside the demuxer.
    #[arg(long = "repeat")]
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
            repeat: self.repeat,
            decode_ahead: self.decode_ahead,
            dav1d_threads: self.dav1d_threads,
        }
    }
}

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
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
    .run()
}

#[derive(Debug, Clone)]
enum Message {
    /// The periodic re-read of position, drops and the bus.
    Tick,
    /// A frame landed on an idle mailbox.
    Frame,
    /// A file finished probing.
    Probed(Arc<Result<Probed, String>>),
    /// The previous engine finished tearing down; build the pending one.
    TornDown,
    /// Open a file.
    Open(PathBuf),
    /// Play/pause.
    TogglePause,
    /// Seek by a number of seconds, relative to where the transport is headed.
    SeekBy(i64),
    /// Step the volume in dB.
    VolumeBy(f32),
    /// Mute/unmute.
    ToggleMute,
    /// Enter or leave fullscreen.
    ToggleFullscreen,
    /// Show or hide the diagnostics panel.
    ToggleStats,
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
    /// A file waiting for the previous engine to finish tearing down.
    pending_open: Option<PathBuf>,

    position: ClockTime,
    paused: bool,
    seek: SeekState,

    /// Volume lives here rather than in the `Gain` handle: the handle knows a
    /// linear factor, the interface thinks in dB, and a mute has to remember
    /// the level it will restore.
    volume_db: f32,
    muted: bool,

    fullscreen: bool,
    show_stats: bool,
    status: String,
    error: Option<String>,

    /// When to quit, for `--exit-after`.
    deadline: Option<Instant>,
}

impl Player {
    fn boot(args: Args) -> (Self, Task<Message>) {
        let player = Self {
            options: args.options(),
            engine: None,
            pending_open: None,
            position: ClockTime::ZERO,
            paused: false,
            seek: SeekState::default(),
            volume_db: 0.0,
            muted: false,
            fullscreen: false,
            show_stats: false,
            status: "no file open".into(),
            error: None,
            deadline: args
                .exit_after
                .map(|s| Instant::now() + Duration::from_secs(s)),
        };
        let task = match args.file {
            Some(path) => Task::done(Message::Open(path)),
            None => Task::none(),
        };
        (player, task)
    }

    fn title(&self) -> String {
        match &self.engine {
            Some(engine) => format!("{} — parallax", engine.source.summary()),
            None => "parallax".into(),
        }
    }

    fn source(&self) -> Option<&SourceDescription> {
        self.engine.as_ref().map(|e| &e.source)
    }

    fn duration(&self) -> ClockTime {
        self.source().map(|s| s.duration).unwrap_or(ClockTime::NONE)
    }

    fn mailbox(&self) -> Option<Arc<FrameMailbox>> {
        self.engine.as_ref().map(|e| Arc::clone(&e.mailbox))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Open(path) => self.open(path),
            Message::TornDown => self.start_pending(),
            Message::Probed(result) => self.build(result),
            Message::Tick => {
                self.poll();
                if self.deadline.is_some_and(|at| Instant::now() >= at) {
                    self.report();
                    return iced::exit();
                }
                Task::none()
            }
            // The subscription's only job is to wake an interface that had
            // nothing queued; the widget reads the mailbox itself.
            Message::Frame => Task::none(),
            Message::TogglePause => {
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
                self.seek_relative(secs);
                Task::none()
            }
            Message::VolumeBy(delta) => {
                self.volume_db = (self.volume_db + delta).clamp(-40.0, 6.0);
                // Reaching for the volume always unmutes: the user wants to
                // hear it.
                self.muted = false;
                self.apply_volume();
                Task::none()
            }
            Message::ToggleMute => {
                self.muted = !self.muted;
                self.apply_volume();
                Task::none()
            }
            Message::ToggleFullscreen => {
                self.fullscreen = !self.fullscreen;
                let mode = if self.fullscreen {
                    iced::window::Mode::Fullscreen
                } else {
                    iced::window::Mode::Windowed
                };
                iced::window::latest().and_then(move |id| iced::window::set_mode(id, mode))
            }
            Message::ToggleStats => {
                self.show_stats = !self.show_stats;
                Task::none()
            }
            Message::Quit => iced::exit(),
        }
    }

    /// Tear the current engine down, then build the new one.
    ///
    /// Two steps rather than one because teardown is bounded-but-slow (it
    /// waits for the tasks to wind down) and the interface must stay live
    /// through it. The engine is taken *now*, so `view()` immediately shows
    /// the placeholder rather than the last frame of the previous file.
    fn open(&mut self, path: PathBuf) -> Task<Message> {
        self.error = None;
        self.status = format!("opening {}…", path.display());
        self.pending_open = Some(path);
        self.reset_transport();

        match self.engine.take() {
            Some(engine) => Task::future(async move {
                engine.teardown().await;
                Message::TornDown
            }),
            None => self.start_pending(),
        }
    }

    fn start_pending(&mut self) -> Task<Message> {
        let Some(path) = self.pending_open.take() else {
            return Task::none();
        };
        // Probing is blocking — an MP4 index parse on a large file is 100 ms
        // and more — so it does not belong on the interface's thread.
        Task::perform(
            async move {
                let result = tokio::task::spawn_blocking(move || {
                    engine::probe(&path).map_err(|e| format!("{e:#}"))
                })
                .await
                .unwrap_or_else(|e| Err(format!("probe panicked: {e}")));
                Arc::new(result)
            },
            Message::Probed,
        )
    }

    /// Build and start the pipeline for a probed file.
    ///
    /// Synchronously, in `update`: `Executor::start` is sync and fast, and
    /// Iced's `tokio` feature runs `update` inside `runtime.enter()`, so the
    /// tasks it spawns find a runtime. Doing it here is what keeps `Engine`
    /// and `PipelineHandle` off the `Send` hook.
    fn build(&mut self, result: Arc<Result<Probed, String>>) -> Task<Message> {
        // The probe result is behind an `Arc` only because an Iced `Message`
        // must be `Clone`; there is exactly one of them.
        let probed = match Arc::try_unwrap(result) {
            Ok(inner) => inner,
            Err(_) => {
                self.error = Some("probe result was cloned".into());
                return Task::none();
            }
        };
        let probed = match probed {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(e);
                self.status = "no file open".into();
                return Task::none();
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
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                self.status = "no file open".into();
            }
        }
        Task::none()
    }

    /// One machine-readable line, for `scripts/decode_matrix.py` and #270.
    fn report(&self) {
        let stats = self
            .engine
            .as_ref()
            .map(|e| e.mailbox.stats())
            .unwrap_or_default();
        let dropped = self.engine.as_ref().map(|e| e.dropped()).unwrap_or(0);
        // `position_ns` and `dropped` stay adjacent and in that order:
        // `scripts/decode_matrix.py` parses exactly that pair, and #270
        // extends the same script rather than replacing it.
        println!(
            "stats: position_ns={} dropped={} presented={} queued={} late={} full={} stale={} clipped={}",
            self.position.to_option().map(|p| p.nanos()).unwrap_or(0),
            dropped,
            stats.presented,
            stats.queued,
            stats.dropped_late,
            stats.dropped_full,
            stats.dropped_stale,
            stats.clipped,
        );
    }

    fn reset_transport(&mut self) {
        self.position = ClockTime::ZERO;
        self.paused = false;
        self.seek = SeekState::default();
    }

    /// Everything the interface re-reads on a tick.
    fn poll(&mut self) {
        let Some(engine) = &mut self.engine else {
            return;
        };
        self.position = engine.position();

        for message in engine.poll_bus() {
            match &message.kind {
                MessageKind::SeekDone { seqnum, .. } => {
                    if self.seek.in_flight.map(|(s, _)| s) == Some(*seqnum) {
                        self.seek.in_flight = None;
                        self.seek.sent = None;
                    }
                }
                MessageKind::Error { error, .. } => self.error = Some(error.clone()),
                MessageKind::Warning { warning, .. } => tracing::warn!("{warning}"),
                _ => {}
            }
        }

        if let Some(reason) = engine.mailbox.end_reason() {
            self.status = format!("{} · {reason:?}", engine.source.summary());
        }

        self.dispatch_seek();
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

        let event = SeekEvent::new_time(ClockTime::from_nanos(target));
        let seqnum = event.seqnum();
        if engine.handle().dispatch_seek(event) {
            self.seek.in_flight = Some((seqnum, target));
            self.seek.sent = Some(now);
        } else {
            self.error = Some("this source cannot seek".into());
        }
        self.seek.pending = None;
        self.seek.due = None;
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
            None => center(text("open a file to start").size(20)).into(),
        };

        stack![
            container(video)
                .width(iced::Fill)
                .height(iced::Fill)
                .style(|_| container::background(iced::Color::BLACK)),
            ui::overlay(self),
        ]
        .into()
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
        Key::Named(Named::Escape) => Some(Message::ToggleFullscreen),
        Key::Character(c) => match c.as_str() {
            "q" => Some(Message::Quit),
            "f" => Some(Message::ToggleFullscreen),
            "m" => Some(Message::ToggleMute),
            "i" => Some(Message::ToggleStats),
            " " => Some(Message::TogglePause),
            _ => None,
        },
        _ => None,
    }
}
