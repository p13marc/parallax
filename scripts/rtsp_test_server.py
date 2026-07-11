#!/usr/bin/env python3
"""Local RTSP test server for the parallax RTSP examples (no VLC needed).

Serves an H.264 test pattern at rtsp://127.0.0.1:8554/stream using
GStreamer's RTSP server via PyGObject. On Fedora the dependencies are:

    sudo dnf install python3-gobject gstreamer1-rtsp-server gstreamer1-plugins-bad-free

Run it, then in another terminal:

    cargo run --example 57_rtsp_capture --features rtsp

Pass a different port or mount point as arguments:

    ./scripts/rtsp_test_server.py [port] [/mount]
"""

import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else "8554"
mount = sys.argv[2] if len(sys.argv) > 2 else "/stream"

Gst.init(None)

server = GstRtspServer.RTSPServer()
server.set_service(port)

factory = GstRtspServer.RTSPMediaFactory()
# openh264enc ships with gstreamer1-plugins-bad; swap in x264enc if you have
# gst-plugins-ugly installed.
factory.set_launch(
    "( videotestsrc is-live=true pattern=smpte "
    "! video/x-raw,width=640,height=360,framerate=25/1 "
    "! openh264enc gop-size=25 "
    "! h264parse ! rtph264pay name=pay0 pt=96 )"
)
factory.set_shared(True)
server.get_mount_points().add_factory(mount, factory)
server.attach(None)

print(f"serving rtsp://127.0.0.1:{port}{mount}  (Ctrl-C to stop)", flush=True)
try:
    GLib.MainLoop().run()
except KeyboardInterrupt:
    pass
