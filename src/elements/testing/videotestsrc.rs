//! VideoTestSrc element for generating video test patterns.
//!
//! Generates various video test patterns for pipeline testing, debugging, and
//! verification. Supports multiple patterns including SMPTE color bars,
//! checkerboard, solid color, and animated patterns.

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::Source;
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Video test pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoPattern {
    /// SMPTE color bars (standard TV test pattern).
    #[default]
    SmpteColorBars,
    /// Checkerboard pattern (configurable square size).
    Checkerboard,
    /// Solid color fill.
    SolidColor,
    /// Moving ball animation.
    MovingBall,
    /// Horizontal color gradient.
    Gradient,
    /// Black screen.
    Black,
    /// White screen.
    White,
    /// Red screen.
    Red,
    /// Green screen.
    Green,
    /// Blue screen.
    Blue,
    /// Circular pattern (concentric rings).
    Circular,
    /// Snow/static noise.
    Snow,
}

/// Pixel format for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelFormat {
    /// RGB with 8 bits per channel (24 bits per pixel).
    #[default]
    Rgb24,
    /// RGBA with 8 bits per channel (32 bits per pixel).
    Rgba32,
    /// BGR with 8 bits per channel (24 bits per pixel).
    Bgr24,
    /// BGRA with 8 bits per channel (32 bits per pixel).
    Bgra32,
}

impl PixelFormat {
    /// Get the number of bytes per pixel.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => 3,
            PixelFormat::Rgba32 | PixelFormat::Bgra32 => 4,
        }
    }

    /// Write a color (r, g, b, a) to the given buffer position.
    fn write_pixel(&self, buf: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
        match self {
            PixelFormat::Rgb24 => {
                buf[0] = r;
                buf[1] = g;
                buf[2] = b;
            }
            PixelFormat::Rgba32 => {
                buf[0] = r;
                buf[1] = g;
                buf[2] = b;
                buf[3] = a;
            }
            PixelFormat::Bgr24 => {
                buf[0] = b;
                buf[1] = g;
                buf[2] = r;
            }
            PixelFormat::Bgra32 => {
                buf[0] = b;
                buf[1] = g;
                buf[2] = r;
                buf[3] = a;
            }
        }
    }
}

/// A source that generates video test pattern frames.
///
/// Useful for testing video pipelines, display elements, and debugging.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{VideoTestSrc, VideoPattern};
///
/// // Generate SMPTE color bars at 1920x1080 @ 30fps
/// let src = VideoTestSrc::new()
///     .with_pattern(VideoPattern::SmpteColorBars)
///     .with_resolution(1920, 1080)
///     .with_framerate(30, 1);
///
/// // Generate 100 frames of a moving ball
/// let src = VideoTestSrc::new()
///     .with_pattern(VideoPattern::MovingBall)
///     .with_resolution(640, 480)
///     .with_num_frames(100);
/// ```
pub struct VideoTestSrc {
    name: String,
    pattern: VideoPattern,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    framerate_num: u32,
    framerate_den: u32,
    num_frames: Option<u64>,
    sequence: u64,
    frames_produced: u64,
    last_produce: Option<Instant>,
    /// Whether to apply framerate limiting (sleep between frames).
    /// Disable this when using in a pipeline where flow is controlled externally.
    live: bool,
    // Pattern-specific state
    solid_color: (u8, u8, u8),
    checker_size: u32,
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    ball_radius: f32,
    rng_state: u64,
}

impl VideoTestSrc {
    /// Create a new video test source with default settings.
    ///
    /// Defaults to SMPTE color bars at 640x480 @ 30fps, non-live (no framerate limiting).
    pub fn new() -> Self {
        Self {
            name: "videotestsrc".to_string(),
            pattern: VideoPattern::default(),
            pixel_format: PixelFormat::default(),
            width: 640,
            height: 480,
            framerate_num: 30,
            framerate_den: 1,
            num_frames: None,
            sequence: 0,
            frames_produced: 0,
            last_produce: None,
            live: false,
            solid_color: (128, 128, 128),
            checker_size: 32,
            ball_x: 100.0,
            ball_y: 100.0,
            ball_vx: 3.0,
            ball_vy: 2.0,
            ball_radius: 30.0,
            rng_state: 0x853c49e6748fea9b,
        }
    }

    /// Set the test pattern.
    pub fn with_pattern(mut self, pattern: VideoPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the pixel format.
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Set the video resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        // Recenter the ball
        self.ball_x = width as f32 / 4.0;
        self.ball_y = height as f32 / 4.0;
        self
    }

    /// Set the framerate as numerator/denominator.
    ///
    /// Common values:
    /// - 30/1 = 30fps
    /// - 60/1 = 60fps
    /// - 24000/1001 ≈ 23.976fps (NTSC film)
    /// - 30000/1001 ≈ 29.97fps (NTSC video)
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        self
    }

    /// Set the number of frames to produce (None = infinite).
    pub fn with_num_frames(mut self, count: u64) -> Self {
        self.num_frames = Some(count);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the solid color (for SolidColor pattern).
    pub fn with_solid_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.solid_color = (r, g, b);
        self
    }

    /// Set the checkerboard square size (for Checkerboard pattern).
    pub fn with_checker_size(mut self, size: u32) -> Self {
        self.checker_size = size;
        self
    }

    /// Set the ball radius (for MovingBall pattern).
    pub fn with_ball_radius(mut self, radius: f32) -> Self {
        self.ball_radius = radius;
        self
    }

    /// Set the ball velocity (for MovingBall pattern).
    pub fn with_ball_velocity(mut self, vx: f32, vy: f32) -> Self {
        self.ball_vx = vx;
        self.ball_vy = vy;
        self
    }

    /// Set the random seed (for Snow pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Enable live mode (real-time framerate limiting).
    ///
    /// When enabled, the source will sleep between frames to maintain the
    /// configured framerate. This is useful for standalone use but should
    /// be disabled when using in a pipeline where flow is controlled by
    /// backpressure or an external clock.
    ///
    /// Default: `false` (no framerate limiting)
    pub fn live(mut self, enabled: bool) -> Self {
        self.live = enabled;
        self
    }

    /// Get the number of frames produced.
    pub fn frames_produced(&self) -> u64 {
        self.frames_produced
    }

    /// Get the frame size in bytes.
    pub fn frame_size(&self) -> usize {
        self.width as usize * self.height as usize * self.pixel_format.bytes_per_pixel()
    }

    /// Get the frame duration.
    pub fn frame_duration(&self) -> ClockTime {
        if self.framerate_num == 0 {
            return ClockTime::NONE;
        }
        ClockTime::from_nanos(
            (self.framerate_den as u64 * 1_000_000_000) / self.framerate_num as u64,
        )
    }

    /// Reset the source.
    pub fn reset(&mut self) {
        self.sequence = 0;
        self.frames_produced = 0;
        self.last_produce = None;
        self.ball_x = self.width as f32 / 4.0;
        self.ball_y = self.height as f32 / 4.0;
    }

    // Simple xorshift64 PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Wait until it's time to produce the next frame (for live mode).
    /// Uses a hybrid approach: sleep for coarse waiting, spin for precision.
    fn wait_for_next_frame(&mut self) {
        // Non-live mode produces immediately
        if !self.live {
            return;
        }

        let frame_duration = self.frame_duration();
        if frame_duration.is_none() {
            return;
        }

        let expected_duration = Duration::from_nanos(frame_duration.nanos());

        let target_time = match self.last_produce {
            None => {
                // First frame - start the timer now
                let now = Instant::now();
                self.last_produce = Some(now);
                return; // Produce first frame immediately
            }
            Some(last) => last + expected_duration,
        };

        let now = Instant::now();
        if now >= target_time {
            // Already past target time, update and produce immediately
            self.last_produce = Some(target_time);
            return;
        }

        let remaining = target_time - now;

        // For waits longer than 2ms, sleep for most of it (leave 1ms for spin)
        if remaining > Duration::from_millis(2) {
            let sleep_duration = remaining - Duration::from_millis(1);
            std::thread::sleep(sleep_duration);
        }

        // Spin-wait for the remaining time for precision
        while Instant::now() < target_time {
            std::hint::spin_loop();
        }

        // Update last_produce to target to avoid drift
        self.last_produce = Some(target_time);
    }

    fn fill_frame(&mut self, data: &mut [u8]) {
        match self.pattern {
            VideoPattern::SmpteColorBars => self.fill_smpte_bars(data),
            VideoPattern::Checkerboard => self.fill_checkerboard(data),
            VideoPattern::SolidColor => self.fill_solid_color(data),
            VideoPattern::MovingBall => self.fill_moving_ball(data),
            VideoPattern::Gradient => self.fill_gradient(data),
            VideoPattern::Black => self.fill_solid(data, 0, 0, 0),
            VideoPattern::White => self.fill_solid(data, 255, 255, 255),
            VideoPattern::Red => self.fill_solid(data, 255, 0, 0),
            VideoPattern::Green => self.fill_solid(data, 0, 255, 0),
            VideoPattern::Blue => self.fill_solid(data, 0, 0, 255),
            VideoPattern::Circular => self.fill_circular(data),
            VideoPattern::Snow => self.fill_snow(data),
        }
    }

    fn fill_solid(&self, data: &mut [u8], r: u8, g: u8, b: u8) {
        let bpp = self.pixel_format.bytes_per_pixel();
        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_solid_color(&self, data: &mut [u8]) {
        let (r, g, b) = self.solid_color;
        self.fill_solid(data, r, g, b);
    }

    fn fill_smpte_bars(&self, data: &mut [u8]) {
        // SMPTE EG 1-1990 color bars
        // Top 2/3: 7 vertical bars (white, yellow, cyan, green, magenta, red, blue)
        // Bottom 1/3: Additional reference patterns

        // 75% amplitude colors (standard SMPTE)
        let bars: [(u8, u8, u8); 7] = [
            (191, 191, 191), // 75% White
            (191, 191, 0),   // 75% Yellow
            (0, 191, 191),   // 75% Cyan
            (0, 191, 0),     // 75% Green
            (191, 0, 191),   // 75% Magenta
            (191, 0, 0),     // 75% Red
            (0, 0, 191),     // 75% Blue
        ];

        let bpp = self.pixel_format.bytes_per_pixel();
        let top_section_height = (self.height * 2) / 3;
        let bar_width = self.width / 7;

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;

                let (r, g, b) = if y < top_section_height {
                    // Top 2/3: color bars
                    let bar_idx = (x / bar_width).min(6) as usize;
                    bars[bar_idx]
                } else {
                    // Bottom 1/3: -I, white, +Q, black, PLUGE pattern
                    let section = (x * 7) / self.width;
                    match section {
                        0 => (0, 33, 76),     // -I (Blue-ish)
                        1 => (255, 255, 255), // 100% White
                        2 => (50, 0, 106),    // +Q (Purple-ish)
                        3 => (0, 0, 0),       // Black
                        4 => {
                            // PLUGE: 3.5% below black
                            if (x % 20) < 10 { (0, 0, 0) } else { (9, 9, 9) }
                        }
                        5 => (0, 0, 0),    // Black
                        6 => (19, 19, 19), // 7.5% above black
                        _ => (0, 0, 0),
                    }
                };

                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_checkerboard(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();
        let size = self.checker_size.max(1);

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let checker_x = x / size;
                let checker_y = y / size;
                let is_white = (checker_x + checker_y) % 2 == 0;

                let (r, g, b) = if is_white { (255, 255, 255) } else { (0, 0, 0) };

                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_moving_ball(&mut self, data: &mut [u8]) {
        // First fill with black background
        self.fill_solid(data, 0, 0, 0);

        let bpp = self.pixel_format.bytes_per_pixel();
        let radius = self.ball_radius;
        let radius_sq = radius * radius;

        // Draw the ball (simple filled circle)
        let min_x = ((self.ball_x - radius).max(0.0)) as u32;
        let max_x = ((self.ball_x + radius).min(self.width as f32 - 1.0)) as u32;
        let min_y = ((self.ball_y - radius).max(0.0)) as u32;
        let max_y = ((self.ball_y + radius).min(self.height as f32 - 1.0)) as u32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - self.ball_x;
                let dy = y as f32 - self.ball_y;
                if dx * dx + dy * dy <= radius_sq {
                    let offset = ((y * self.width + x) as usize) * bpp;
                    // Color the ball based on frame number for visual interest
                    let hue = (self.sequence * 5) % 360;
                    let (r, g, b) = hsv_to_rgb(hue as f32, 1.0, 1.0);
                    self.pixel_format
                        .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
                }
            }
        }

        // Update ball position for next frame
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        // Bounce off walls
        if self.ball_x - radius < 0.0 || self.ball_x + radius >= self.width as f32 {
            self.ball_vx = -self.ball_vx;
            self.ball_x = self.ball_x.clamp(radius, self.width as f32 - radius - 1.0);
        }
        if self.ball_y - radius < 0.0 || self.ball_y + radius >= self.height as f32 {
            self.ball_vy = -self.ball_vy;
            self.ball_y = self.ball_y.clamp(radius, self.height as f32 - radius - 1.0);
        }
    }

    fn fill_gradient(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                // Horizontal gradient through hue spectrum
                let hue = (x as f32 / self.width as f32) * 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_circular(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();
        let cx = self.width as f32 / 2.0;
        let cy = self.height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let normalized = dist / max_dist;

                // Create concentric rings
                let ring = ((normalized * 10.0) as u32) % 2;
                let intensity = if ring == 0 { 255 } else { 0 };

                self.pixel_format.write_pixel(
                    &mut data[offset..offset + bpp],
                    intensity,
                    intensity,
                    intensity,
                    255,
                );
            }
        }
    }

    fn fill_snow(&mut self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let intensity = (self.next_random() & 0xFF) as u8;
                self.pixel_format.write_pixel(
                    &mut data[offset..offset + bpp],
                    intensity,
                    intensity,
                    intensity,
                    255,
                );
            }
        }
    }
}

impl Default for VideoTestSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for VideoTestSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        // Check frame limit
        if let Some(max) = self.num_frames {
            if self.sequence >= max {
                return Ok(None);
            }
        }

        // Wait for next frame time (only in live mode)
        self.wait_for_next_frame();

        // Create frame buffer
        let frame_size = self.frame_size();
        let segment = Arc::new(HeapSegment::new(frame_size)?);
        let ptr = segment.as_mut_ptr().unwrap();
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, frame_size) };

        self.fill_frame(data);

        let handle = MemoryHandle::from_segment_with_len(segment, frame_size);

        // Calculate PTS based on frame number and framerate
        let pts = ClockTime::from_nanos(
            (self.sequence * self.framerate_den as u64 * 1_000_000_000) / self.framerate_num as u64,
        );

        let metadata = Metadata::from_sequence(self.sequence)
            .with_pts(pts)
            .with_duration(self.frame_duration())
            .keyframe(); // Video test patterns are always keyframes

        self.sequence += 1;
        self.frames_produced += 1;

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Convert HSV to RGB color.
///
/// H: 0-360, S: 0-1, V: 0-1
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// An async video test source that uses tokio timers for precise framerate control.
///
/// This is the async version of [`VideoTestSrc`] that uses `tokio::time::Interval`
/// for precise, non-blocking framerate limiting. This is ideal for use in async
/// pipelines where blocking the executor is undesirable.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{AsyncVideoTestSrc, VideoPattern};
///
/// // Generate SMPTE color bars at 30fps in live mode
/// let mut src = AsyncVideoTestSrc::new()
///     .with_pattern(VideoPattern::SmpteColorBars)
///     .with_resolution(1920, 1080)
///     .with_framerate(30, 1)
///     .live(true);
///
/// // In an async context
/// while let Some(buffer) = src.produce().await? {
///     // Process frame...
/// }
/// ```
pub struct AsyncVideoTestSrc {
    name: String,
    pattern: VideoPattern,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    framerate_num: u32,
    framerate_den: u32,
    num_frames: Option<u64>,
    sequence: u64,
    frames_produced: u64,
    /// Whether to apply framerate limiting using tokio timer.
    live: bool,
    /// Tokio interval for framerate timing (lazily initialized).
    interval: Option<tokio::time::Interval>,
    // Pattern-specific state
    solid_color: (u8, u8, u8),
    checker_size: u32,
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    ball_radius: f32,
    rng_state: u64,
}

impl AsyncVideoTestSrc {
    /// Create a new async video test source with default settings.
    ///
    /// Defaults to SMPTE color bars at 640x480 @ 30fps, non-live (no framerate limiting).
    pub fn new() -> Self {
        Self {
            name: "async-videotestsrc".to_string(),
            pattern: VideoPattern::default(),
            pixel_format: PixelFormat::default(),
            width: 640,
            height: 480,
            framerate_num: 30,
            framerate_den: 1,
            num_frames: None,
            sequence: 0,
            frames_produced: 0,
            live: false,
            interval: None,
            solid_color: (128, 128, 128),
            checker_size: 32,
            ball_x: 100.0,
            ball_y: 100.0,
            ball_vx: 3.0,
            ball_vy: 2.0,
            ball_radius: 30.0,
            rng_state: 0x853c49e6748fea9b,
        }
    }

    /// Set the test pattern.
    pub fn with_pattern(mut self, pattern: VideoPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the pixel format.
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Set the video resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self.ball_x = width as f32 / 4.0;
        self.ball_y = height as f32 / 4.0;
        self
    }

    /// Set the framerate as numerator/denominator.
    ///
    /// Common values:
    /// - 30/1 = 30fps
    /// - 60/1 = 60fps
    /// - 24000/1001 ≈ 23.976fps (NTSC film)
    /// - 30000/1001 ≈ 29.97fps (NTSC video)
    pub fn with_framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate_num = num;
        self.framerate_den = den;
        // Reset interval so it gets recreated with new framerate
        self.interval = None;
        self
    }

    /// Set the number of frames to produce (None = infinite).
    pub fn with_num_frames(mut self, count: u64) -> Self {
        self.num_frames = Some(count);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the solid color (for SolidColor pattern).
    pub fn with_solid_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.solid_color = (r, g, b);
        self
    }

    /// Set the checkerboard square size (for Checkerboard pattern).
    pub fn with_checker_size(mut self, size: u32) -> Self {
        self.checker_size = size;
        self
    }

    /// Set the ball radius (for MovingBall pattern).
    pub fn with_ball_radius(mut self, radius: f32) -> Self {
        self.ball_radius = radius;
        self
    }

    /// Set the ball velocity (for MovingBall pattern).
    pub fn with_ball_velocity(mut self, vx: f32, vy: f32) -> Self {
        self.ball_vx = vx;
        self.ball_vy = vy;
        self
    }

    /// Set the random seed (for Snow pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Enable live mode (real-time framerate limiting using tokio timer).
    ///
    /// When enabled, the source will use `tokio::time::Interval` to maintain
    /// the configured framerate. This provides precise, non-blocking timing
    /// that works well in async pipelines.
    ///
    /// Default: `false` (no framerate limiting)
    pub fn live(mut self, enabled: bool) -> Self {
        self.live = enabled;
        self
    }

    /// Get the number of frames produced.
    pub fn frames_produced(&self) -> u64 {
        self.frames_produced
    }

    /// Get the frame size in bytes.
    pub fn frame_size(&self) -> usize {
        self.width as usize * self.height as usize * self.pixel_format.bytes_per_pixel()
    }

    /// Get the frame duration.
    pub fn frame_duration(&self) -> ClockTime {
        if self.framerate_num == 0 {
            return ClockTime::NONE;
        }
        ClockTime::from_nanos(
            (self.framerate_den as u64 * 1_000_000_000) / self.framerate_num as u64,
        )
    }

    /// Reset the source.
    pub fn reset(&mut self) {
        self.sequence = 0;
        self.frames_produced = 0;
        self.interval = None;
        self.ball_x = self.width as f32 / 4.0;
        self.ball_y = self.height as f32 / 4.0;
    }

    // Simple xorshift64 PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Ensure the interval is initialized for live mode.
    fn ensure_interval(&mut self) {
        if self.live && self.interval.is_none() {
            let frame_duration = self.frame_duration();
            if !frame_duration.is_none() {
                let duration = Duration::from_nanos(frame_duration.nanos());
                let mut interval = tokio::time::interval(duration);
                // Use MissedTickBehavior::Skip to avoid catching up if we fall behind
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                self.interval = Some(interval);
            }
        }
    }

    fn fill_frame(&mut self, data: &mut [u8]) {
        match self.pattern {
            VideoPattern::SmpteColorBars => self.fill_smpte_bars(data),
            VideoPattern::Checkerboard => self.fill_checkerboard(data),
            VideoPattern::SolidColor => self.fill_solid_color(data),
            VideoPattern::MovingBall => self.fill_moving_ball(data),
            VideoPattern::Gradient => self.fill_gradient(data),
            VideoPattern::Black => self.fill_solid(data, 0, 0, 0),
            VideoPattern::White => self.fill_solid(data, 255, 255, 255),
            VideoPattern::Red => self.fill_solid(data, 255, 0, 0),
            VideoPattern::Green => self.fill_solid(data, 0, 255, 0),
            VideoPattern::Blue => self.fill_solid(data, 0, 0, 255),
            VideoPattern::Circular => self.fill_circular(data),
            VideoPattern::Snow => self.fill_snow(data),
        }
    }

    fn fill_solid(&self, data: &mut [u8], r: u8, g: u8, b: u8) {
        let bpp = self.pixel_format.bytes_per_pixel();
        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_solid_color(&self, data: &mut [u8]) {
        let (r, g, b) = self.solid_color;
        self.fill_solid(data, r, g, b);
    }

    fn fill_smpte_bars(&self, data: &mut [u8]) {
        let bars: [(u8, u8, u8); 7] = [
            (191, 191, 191),
            (191, 191, 0),
            (0, 191, 191),
            (0, 191, 0),
            (191, 0, 191),
            (191, 0, 0),
            (0, 0, 191),
        ];

        let bpp = self.pixel_format.bytes_per_pixel();
        let top_section_height = (self.height * 2) / 3;
        let bar_width = self.width / 7;

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;

                let (r, g, b) = if y < top_section_height {
                    let bar_idx = (x / bar_width).min(6) as usize;
                    bars[bar_idx]
                } else {
                    let section = (x * 7) / self.width;
                    match section {
                        0 => (0, 33, 76),
                        1 => (255, 255, 255),
                        2 => (50, 0, 106),
                        3 => (0, 0, 0),
                        4 => {
                            if (x % 20) < 10 {
                                (0, 0, 0)
                            } else {
                                (9, 9, 9)
                            }
                        }
                        5 => (0, 0, 0),
                        6 => (19, 19, 19),
                        _ => (0, 0, 0),
                    }
                };

                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_checkerboard(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();
        let size = self.checker_size.max(1);

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let checker_x = x / size;
                let checker_y = y / size;
                let is_white = (checker_x + checker_y) % 2 == 0;

                let (r, g, b) = if is_white { (255, 255, 255) } else { (0, 0, 0) };

                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_moving_ball(&mut self, data: &mut [u8]) {
        self.fill_solid(data, 0, 0, 0);

        let bpp = self.pixel_format.bytes_per_pixel();
        let radius = self.ball_radius;
        let radius_sq = radius * radius;

        let min_x = ((self.ball_x - radius).max(0.0)) as u32;
        let max_x = ((self.ball_x + radius).min(self.width as f32 - 1.0)) as u32;
        let min_y = ((self.ball_y - radius).max(0.0)) as u32;
        let max_y = ((self.ball_y + radius).min(self.height as f32 - 1.0)) as u32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - self.ball_x;
                let dy = y as f32 - self.ball_y;
                if dx * dx + dy * dy <= radius_sq {
                    let offset = ((y * self.width + x) as usize) * bpp;
                    let hue = (self.sequence * 5) % 360;
                    let (r, g, b) = hsv_to_rgb(hue as f32, 1.0, 1.0);
                    self.pixel_format
                        .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
                }
            }
        }

        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        if self.ball_x - radius < 0.0 || self.ball_x + radius >= self.width as f32 {
            self.ball_vx = -self.ball_vx;
            self.ball_x = self.ball_x.clamp(radius, self.width as f32 - radius - 1.0);
        }
        if self.ball_y - radius < 0.0 || self.ball_y + radius >= self.height as f32 {
            self.ball_vy = -self.ball_vy;
            self.ball_y = self.ball_y.clamp(radius, self.height as f32 - radius - 1.0);
        }
    }

    fn fill_gradient(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let hue = (x as f32 / self.width as f32) * 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                self.pixel_format
                    .write_pixel(&mut data[offset..offset + bpp], r, g, b, 255);
            }
        }
    }

    fn fill_circular(&self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();
        let cx = self.width as f32 / 2.0;
        let cy = self.height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let normalized = dist / max_dist;

                let ring = ((normalized * 10.0) as u32) % 2;
                let intensity = if ring == 0 { 255 } else { 0 };

                self.pixel_format.write_pixel(
                    &mut data[offset..offset + bpp],
                    intensity,
                    intensity,
                    intensity,
                    255,
                );
            }
        }
    }

    fn fill_snow(&mut self, data: &mut [u8]) {
        let bpp = self.pixel_format.bytes_per_pixel();

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) as usize) * bpp;
                let intensity = (self.next_random() & 0xFF) as u8;
                self.pixel_format.write_pixel(
                    &mut data[offset..offset + bpp],
                    intensity,
                    intensity,
                    intensity,
                    255,
                );
            }
        }
    }
}

impl Default for AsyncVideoTestSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::element::AsyncSource for AsyncVideoTestSrc {
    async fn produce(&mut self) -> Result<Option<Buffer>> {
        // Check frame limit
        if let Some(max) = self.num_frames {
            if self.sequence >= max {
                return Ok(None);
            }
        }

        // Wait for next frame time using tokio interval (only in live mode)
        if self.live {
            self.ensure_interval();
            if let Some(ref mut interval) = self.interval {
                interval.tick().await;
            }
        }

        // Create frame buffer
        let frame_size = self.frame_size();
        let segment = Arc::new(HeapSegment::new(frame_size)?);
        let ptr = segment.as_mut_ptr().unwrap();
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, frame_size) };

        self.fill_frame(data);

        let handle = MemoryHandle::from_segment_with_len(segment, frame_size);

        // Calculate PTS based on frame number and framerate
        let pts = ClockTime::from_nanos(
            (self.sequence * self.framerate_den as u64 * 1_000_000_000) / self.framerate_num as u64,
        );

        let metadata = Metadata::from_sequence(self.sequence)
            .with_pts(pts)
            .with_duration(self.frame_duration())
            .keyframe();

        self.sequence += 1;
        self.frames_produced += 1;

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_videotestsrc_default() {
        let src = VideoTestSrc::new();
        assert_eq!(src.width, 640);
        assert_eq!(src.height, 480);
        assert_eq!(src.framerate_num, 30);
        assert_eq!(src.framerate_den, 1);
        assert_eq!(src.pattern, VideoPattern::SmpteColorBars);
    }

    #[test]
    fn test_videotestsrc_frame_size() {
        let src = VideoTestSrc::new()
            .with_resolution(100, 100)
            .with_pixel_format(PixelFormat::Rgb24);
        assert_eq!(src.frame_size(), 100 * 100 * 3);

        let src = VideoTestSrc::new()
            .with_resolution(100, 100)
            .with_pixel_format(PixelFormat::Rgba32);
        assert_eq!(src.frame_size(), 100 * 100 * 4);
    }

    #[test]
    fn test_videotestsrc_frame_duration() {
        let src = VideoTestSrc::new().with_framerate(30, 1);
        let duration = src.frame_duration();
        assert_eq!(duration.millis(), 33); // ~33.3ms

        let src = VideoTestSrc::new().with_framerate(60, 1);
        let duration = src.frame_duration();
        assert_eq!(duration.millis(), 16); // ~16.6ms
    }

    #[test]
    fn test_videotestsrc_produce_smpte() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::SmpteColorBars)
            .with_resolution(100, 100)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.len(), 100 * 100 * 3);
        assert!(buf.metadata().is_keyframe());
    }

    #[test]
    fn test_videotestsrc_produce_checkerboard() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::Checkerboard)
            .with_checker_size(10)
            .with_resolution(100, 100)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        // First pixel should be white (255, 255, 255)
        assert_eq!(data[0], 255);
        assert_eq!(data[1], 255);
        assert_eq!(data[2], 255);

        // Pixel at (10, 0) should be black (0, 0, 0)
        let offset = 10 * 3;
        assert_eq!(data[offset], 0);
        assert_eq!(data[offset + 1], 0);
        assert_eq!(data[offset + 2], 0);
    }

    #[test]
    fn test_videotestsrc_produce_solid_color() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::SolidColor)
            .with_solid_color(128, 64, 32)
            .with_resolution(10, 10)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        // Every pixel should be (128, 64, 32)
        for i in 0..10 * 10 {
            assert_eq!(data[i * 3], 128);
            assert_eq!(data[i * 3 + 1], 64);
            assert_eq!(data[i * 3 + 2], 32);
        }
    }

    #[test]
    fn test_videotestsrc_produce_black() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::Black)
            .with_resolution(10, 10)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_videotestsrc_produce_white() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::White)
            .with_resolution(10, 10)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        assert!(data.iter().all(|&b| b == 255));
    }

    #[test]
    fn test_videotestsrc_moving_ball() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::MovingBall)
            .with_resolution(100, 100)
            .with_ball_radius(10.0)
            .with_num_frames(10);

        // Produce multiple frames to test ball movement
        for _ in 0..10 {
            let buf = src.produce().unwrap();
            assert!(buf.is_some());
        }

        // Should return None after 10 frames
        assert!(src.produce().unwrap().is_none());
    }

    #[test]
    fn test_videotestsrc_num_frames() {
        let mut src = VideoTestSrc::new()
            .with_resolution(10, 10)
            .with_num_frames(3);

        assert!(src.produce().unwrap().is_some());
        assert!(src.produce().unwrap().is_some());
        assert!(src.produce().unwrap().is_some());
        assert!(src.produce().unwrap().is_none());

        assert_eq!(src.frames_produced(), 3);
    }

    #[test]
    fn test_videotestsrc_sequence_and_pts() {
        let mut src = VideoTestSrc::new()
            .with_resolution(10, 10)
            .with_framerate(30, 1)
            .with_num_frames(3);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 0);
        assert_eq!(buf.metadata().pts.nanos(), 0);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 1);
        // At 30fps, frame 1 is at ~33.3ms
        assert!(buf.metadata().pts.millis() >= 33 && buf.metadata().pts.millis() <= 34);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 2);
        assert!(buf.metadata().pts.millis() >= 66 && buf.metadata().pts.millis() <= 67);
    }

    #[test]
    fn test_videotestsrc_reset() {
        let mut src = VideoTestSrc::new()
            .with_resolution(10, 10)
            .with_num_frames(2);

        src.produce().unwrap();
        src.produce().unwrap();
        assert!(src.produce().unwrap().is_none());

        src.reset();

        assert!(src.produce().unwrap().is_some());
        assert_eq!(src.frames_produced(), 1);
    }

    #[test]
    fn test_videotestsrc_rgba_format() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::Red)
            .with_pixel_format(PixelFormat::Rgba32)
            .with_resolution(10, 10)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        // RGBA: red should be (255, 0, 0, 255)
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 255); // A
    }

    #[test]
    fn test_videotestsrc_bgr_format() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::Red)
            .with_pixel_format(PixelFormat::Bgr24)
            .with_resolution(10, 10)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        // BGR: red should be (0, 0, 255)
        assert_eq!(data[0], 0); // B
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 255); // R
    }

    #[test]
    fn test_videotestsrc_snow_reproducible() {
        let mut src1 = VideoTestSrc::new()
            .with_pattern(VideoPattern::Snow)
            .with_resolution(10, 10)
            .with_seed(12345)
            .with_num_frames(1);

        let mut src2 = VideoTestSrc::new()
            .with_pattern(VideoPattern::Snow)
            .with_resolution(10, 10)
            .with_seed(12345)
            .with_num_frames(1);

        let buf1 = src1.produce().unwrap().unwrap();
        let buf2 = src2.produce().unwrap().unwrap();

        assert_eq!(buf1.as_bytes(), buf2.as_bytes());
    }

    #[test]
    fn test_videotestsrc_gradient() {
        let mut src = VideoTestSrc::new()
            .with_pattern(VideoPattern::Gradient)
            .with_resolution(360, 1)
            .with_num_frames(1);

        let buf = src.produce().unwrap().unwrap();
        let data = buf.as_bytes();

        // First pixel should be red-ish (hue = 0)
        assert!(data[0] > 200); // R high
        assert!(data[1] < 50); // G low
        assert!(data[2] < 50); // B low

        // Pixel at x=60 should be yellow-ish (hue = 60)
        let offset = 60 * 3;
        assert!(data[offset] > 200); // R high
        assert!(data[offset + 1] > 200); // G high
        assert!(data[offset + 2] < 50); // B low
    }

    #[test]
    fn test_videotestsrc_with_name() {
        let src = VideoTestSrc::new().with_name("my-video-test");
        assert_eq!(src.name(), "my-video-test");
    }

    #[test]
    fn test_hsv_to_rgb() {
        // Red
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!((r, g, b), (255, 0, 0));

        // Green
        let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
        assert_eq!((r, g, b), (0, 255, 0));

        // Blue
        let (r, g, b) = hsv_to_rgb(240.0, 1.0, 1.0);
        assert_eq!((r, g, b), (0, 0, 255));

        // White
        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 1.0);
        assert_eq!((r, g, b), (255, 255, 255));

        // Black
        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 0.0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    // =========================================================================
    // AsyncVideoTestSrc tests
    // =========================================================================

    #[test]
    fn test_async_videotestsrc_default() {
        let src = AsyncVideoTestSrc::new();
        assert_eq!(src.width, 640);
        assert_eq!(src.height, 480);
        assert_eq!(src.framerate_num, 30);
        assert_eq!(src.framerate_den, 1);
        assert_eq!(src.pattern, VideoPattern::SmpteColorBars);
        assert!(!src.live);
    }

    #[test]
    fn test_async_videotestsrc_frame_size() {
        let src = AsyncVideoTestSrc::new()
            .with_resolution(100, 100)
            .with_pixel_format(PixelFormat::Rgb24);
        assert_eq!(src.frame_size(), 100 * 100 * 3);

        let src = AsyncVideoTestSrc::new()
            .with_resolution(100, 100)
            .with_pixel_format(PixelFormat::Rgba32);
        assert_eq!(src.frame_size(), 100 * 100 * 4);
    }

    #[tokio::test]
    async fn test_async_videotestsrc_produce_smpte() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::SmpteColorBars)
            .with_resolution(100, 100)
            .with_num_frames(1);

        let buf = src.produce().await.unwrap().unwrap();
        assert_eq!(buf.len(), 100 * 100 * 3);
        assert!(buf.metadata().is_keyframe());
    }

    #[tokio::test]
    async fn test_async_videotestsrc_produce_checkerboard() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::Checkerboard)
            .with_checker_size(10)
            .with_resolution(100, 100)
            .with_num_frames(1);

        let buf = src.produce().await.unwrap().unwrap();
        let data = buf.as_bytes();

        // First pixel should be white (255, 255, 255)
        assert_eq!(data[0], 255);
        assert_eq!(data[1], 255);
        assert_eq!(data[2], 255);

        // Pixel at (10, 0) should be black (0, 0, 0)
        let offset = 10 * 3;
        assert_eq!(data[offset], 0);
        assert_eq!(data[offset + 1], 0);
        assert_eq!(data[offset + 2], 0);
    }

    #[tokio::test]
    async fn test_async_videotestsrc_num_frames() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_resolution(10, 10)
            .with_num_frames(3);

        assert!(src.produce().await.unwrap().is_some());
        assert!(src.produce().await.unwrap().is_some());
        assert!(src.produce().await.unwrap().is_some());
        assert!(src.produce().await.unwrap().is_none());

        assert_eq!(src.frames_produced(), 3);
    }

    #[tokio::test]
    async fn test_async_videotestsrc_sequence_and_pts() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_resolution(10, 10)
            .with_framerate(30, 1)
            .with_num_frames(3);

        let buf = src.produce().await.unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 0);
        assert_eq!(buf.metadata().pts.nanos(), 0);

        let buf = src.produce().await.unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 1);
        assert!(buf.metadata().pts.millis() >= 33 && buf.metadata().pts.millis() <= 34);

        let buf = src.produce().await.unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 2);
        assert!(buf.metadata().pts.millis() >= 66 && buf.metadata().pts.millis() <= 67);
    }

    #[tokio::test]
    async fn test_async_videotestsrc_reset() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_resolution(10, 10)
            .with_num_frames(2);

        src.produce().await.unwrap();
        src.produce().await.unwrap();
        assert!(src.produce().await.unwrap().is_none());

        src.reset();

        assert!(src.produce().await.unwrap().is_some());
        assert_eq!(src.frames_produced(), 1);
    }

    #[tokio::test]
    async fn test_async_videotestsrc_live_mode_timing() {
        use crate::element::AsyncSource;
        use std::time::Instant;

        // Test that live mode respects framerate timing
        let mut src = AsyncVideoTestSrc::new()
            .with_resolution(10, 10)
            .with_framerate(60, 1) // 60fps = ~16.6ms per frame
            .with_num_frames(3)
            .live(true);

        let start = Instant::now();

        // Produce 3 frames
        src.produce().await.unwrap();
        src.produce().await.unwrap();
        src.produce().await.unwrap();

        let elapsed = start.elapsed();

        // Should take at least 2 frame intervals (~33ms for 60fps)
        // Using a generous lower bound to account for timing variations
        assert!(
            elapsed >= Duration::from_millis(25),
            "Expected at least 25ms for 3 frames at 60fps, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_async_videotestsrc_non_live_mode_fast() {
        use crate::element::AsyncSource;
        use std::time::Instant;

        // Test that non-live mode produces frames as fast as possible
        let mut src = AsyncVideoTestSrc::new()
            .with_resolution(10, 10)
            .with_framerate(1, 1) // 1fps = 1000ms per frame (but we're not live)
            .with_num_frames(10)
            .live(false);

        let start = Instant::now();

        // Produce 10 frames
        for _ in 0..10 {
            src.produce().await.unwrap();
        }

        let elapsed = start.elapsed();

        // Non-live mode should produce frames very quickly (< 100ms total)
        assert!(
            elapsed < Duration::from_millis(100),
            "Non-live mode should be fast, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_async_videotestsrc_moving_ball() {
        use crate::element::AsyncSource;

        let mut src = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::MovingBall)
            .with_resolution(100, 100)
            .with_ball_radius(10.0)
            .with_num_frames(10);

        // Produce multiple frames to test ball movement
        for _ in 0..10 {
            let buf = src.produce().await.unwrap();
            assert!(buf.is_some());
        }

        // Should return None after 10 frames
        assert!(src.produce().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_async_videotestsrc_snow_reproducible() {
        use crate::element::AsyncSource;

        let mut src1 = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::Snow)
            .with_resolution(10, 10)
            .with_seed(12345)
            .with_num_frames(1);

        let mut src2 = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::Snow)
            .with_resolution(10, 10)
            .with_seed(12345)
            .with_num_frames(1);

        let buf1 = src1.produce().await.unwrap().unwrap();
        let buf2 = src2.produce().await.unwrap().unwrap();

        assert_eq!(buf1.as_bytes(), buf2.as_bytes());
    }

    #[test]
    fn test_async_videotestsrc_with_name() {
        let src = AsyncVideoTestSrc::new().with_name("my-async-video-test");
        assert_eq!(src.name, "my-async-video-test");
    }

    #[test]
    fn test_async_videotestsrc_builder_methods() {
        let src = AsyncVideoTestSrc::new()
            .with_pattern(VideoPattern::Gradient)
            .with_pixel_format(PixelFormat::Rgba32)
            .with_resolution(1920, 1080)
            .with_framerate(60, 1)
            .with_num_frames(100)
            .with_solid_color(255, 128, 64)
            .with_checker_size(16)
            .with_ball_radius(50.0)
            .with_ball_velocity(5.0, 3.0)
            .with_seed(999)
            .live(true);

        assert_eq!(src.pattern, VideoPattern::Gradient);
        assert_eq!(src.pixel_format, PixelFormat::Rgba32);
        assert_eq!(src.width, 1920);
        assert_eq!(src.height, 1080);
        assert_eq!(src.framerate_num, 60);
        assert_eq!(src.framerate_den, 1);
        assert_eq!(src.num_frames, Some(100));
        assert_eq!(src.solid_color, (255, 128, 64));
        assert_eq!(src.checker_size, 16);
        assert_eq!(src.ball_radius, 50.0);
        assert_eq!(src.ball_vx, 5.0);
        assert_eq!(src.ball_vy, 3.0);
        assert_eq!(src.rng_state, 999);
        assert!(src.live);
    }
}
