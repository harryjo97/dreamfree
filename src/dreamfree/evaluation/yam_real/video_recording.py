"""Continuous per-camera MP4 recording from ABC's existing RGB publishers."""

import threading
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np

if __package__:
    from .common import CAMERAS
else:
    from common import CAMERAS


TIME_BASE = Fraction(1, 1_000_000)


def _video_stream(container):
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height = 640, 480
    stream.pix_fmt = "yuv420p"
    stream.time_base = stream.codec_context.time_base = TIME_BASE
    stream.codec_context.thread_count = 1
    stream.codec_context.max_b_frames = 0
    stream.options = {"preset": "ultrafast", "crf": "23", "tune": "zerolatency"}
    return stream


def check_video_backend():
    """Check the actual encoder/muxer in memory before any hardware starts."""
    try:
        import av

        with av.open(BytesIO(), "w", format="mp4") as container:
            stream = _video_stream(container)
            frame = av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = 0, TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        raise RuntimeError(
            "YAM video recording requires PyAV (av) with a working libx264 MP4 encoder. "
            "Install it in the workstation environment or explicitly use --no-record-video."
        ) from exc


class CameraVideoRecorder:
    """Own sockets and encoders in one worker; surface failures to the action loop."""

    def __init__(
        self, cameras, output_dir, episode_index, *, startup_timeout=30.0, frame_timeout=2.0
    ):
        self.cameras = cameras
        self.output_dir = Path(output_dir)
        self.startup_timeout, self.frame_timeout = startup_timeout, frame_timeout
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self._started = False
        self._status = "not_started"
        self._records = {
            name: dict(
                path=f"episode_{episode_index:03d}_{name}.mp4",
                frames=0,
                first_timestamp=None,
                last_timestamp=None,
                largest_gap_seconds=0.0,
            )
            for name in CAMERAS
        }
        for record in self._records.values():
            if (self.output_dir / record["path"]).exists():
                raise FileExistsError(f"Video already exists: {self.output_dir / record['path']}")
        self._thread = threading.Thread(target=self._record, name="yam_video", daemon=True)

    def start(self):
        self._started = True
        self._status = "recording"
        self._thread.start()
        if not self._ready.wait(self.startup_timeout + 1):
            self._error = RuntimeError(
                "YAM video recorder did not become ready before the startup timeout."
            )
        self.check()
        print(
            f"Recording camera videos in {self.output_dir}: "
            + ", ".join(record["path"] for record in self._records.values()),
            flush=True,
        )

    def check(self):
        if self._error is not None:
            raise RuntimeError(f"YAM video recording failed: {self._error}") from self._error
        if self._started and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("YAM video recorder stopped unexpectedly.")

    def close(self):
        self._stop.set()
        if self._started:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                self._status = "failed"
                self._error = RuntimeError(
                    "YAM video recorder did not finish writing MP4 files within 3 seconds."
                )
        self.check()

    def metadata(self):
        with self._lock:
            return dict(
                enabled=True,
                status=self._status,
                codec="libx264",
                pixel_format="yuv420p",
                frame_size_hw=[480, 640],
                nominal_fps=30,
                timestamp_clock="time.perf_counter",
                video_time_origin="per-camera first_timestamp",
                cameras={name: dict(record) for name, record in self._records.items()},
                error=str(self._error) if self._error is not None else None,
            )

    def _record(self):
        context = None
        sockets, outputs = {}, {}
        try:
            import av
            import deploy.robot.communication as comms
            import zmq

            context = zmq.Context()
            # Bound the recording backlog without conflating it to policy-rate snapshots.
            context.setsockopt(zmq.RCVHWM, 60)
            poller = zmq.Poller()
            for name in CAMERAS:
                sockets[name] = comms.create_subscriber(
                    context, self.cameras[name].socket or f"{name}_rgb"
                )
                poller.register(sockets[name], zmq.POLLIN)
                container = av.open(
                    str(self.output_dir / self._records[name]["path"]), "w", format="mp4"
                )
                outputs[name] = (container, None)
                outputs[name] = (container, _video_stream(container))
            started = time.perf_counter()
            while not self._stop.is_set():
                available = dict(poller.poll(50))
                for name, socket in sockets.items():
                    record = self._records[name]
                    if socket in available:
                        rgb, extras = comms.subscribe(
                            socket, timeout_ms=0, topic_label=f"video:{name}"
                        )
                        timestamp = extras.get("timestamp")
                        if (
                            not isinstance(rgb, np.ndarray)
                            or rgb.dtype != np.uint8
                            or rgb.shape != (480, 640, 3)
                        ):
                            raise ValueError(
                                f"Video {name} requires uint8 RGB frames with shape (480,640,3)."
                            )
                        if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
                            raise ValueError(f"Video {name} is missing a finite camera timestamp.")
                        last = record["last_timestamp"]
                        if last is not None and timestamp <= last:
                            raise ValueError(f"Video {name} camera timestamp did not advance.")
                        if time.perf_counter() - timestamp > self.frame_timeout:
                            raise RuntimeError(
                                f"Video {name} frame is stale; camera or encoder cannot keep up."
                            )
                        first = (
                            timestamp
                            if record["first_timestamp"] is None
                            else record["first_timestamp"]
                        )
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        # Preserve elapsed camera time, including gaps and long policy RPC calls.
                        frame.pts = round((timestamp - first) / TIME_BASE)
                        frame.time_base = TIME_BASE
                        container, stream = outputs[name]
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        with self._lock:
                            record.update(
                                frames=record["frames"] + 1,
                                first_timestamp=first,
                                last_timestamp=timestamp,
                                largest_gap_seconds=max(
                                    record["largest_gap_seconds"],
                                    timestamp - last if last is not None else 0.0,
                                ),
                            )
                    now = time.perf_counter()
                    if record["last_timestamp"] is None:
                        if now - started > self.startup_timeout:
                            raise RuntimeError(
                                f"Video {name} received no frames within {self.startup_timeout:g} seconds."
                            )
                    elif now - record["last_timestamp"] > self.frame_timeout:
                        raise RuntimeError(
                            f"Video {name} camera stalled for more than {self.frame_timeout:g} seconds."
                        )
                if all(record["frames"] for record in self._records.values()):
                    self._ready.set()
        except BaseException as exc:
            self._error = exc
        finally:
            # Flush every view, even if one camera or encoder failed.
            for container, stream in outputs.values():
                try:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                except Exception as exc:
                    if self._error is None:
                        self._error = exc
                finally:
                    try:
                        container.close()
                    except Exception as exc:
                        if self._error is None:
                            self._error = exc
            for socket in sockets.values():
                socket.close(linger=0)
            if context is not None:
                context.term()
            self._status = (
                "failed"
                if self._error is not None
                else "complete"
                if all(record["frames"] for record in self._records.values())
                else "incomplete"
            )
            self._ready.set()
