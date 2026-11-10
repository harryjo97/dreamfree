#!/usr/bin/env python3
"""Preview ABC's top/left/right RGB cameras without opening arms or a policy connection."""

import argparse
import os
import time

import numpy as np

if __package__:
    from .common import CAMERAS
    from .hardware import add_station_arguments, launch_nodes, load_profile, station_lock
else:
    from common import CAMERAS
    from hardware import add_station_arguments, launch_nodes, load_profile, station_lock


def validate_frame(name, frame, timestamp, previous_timestamp, max_age):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
        raise ValueError(
            f"Camera {name}: expected uint8 RGB (480,640,3), got "
            f"{getattr(frame, 'shape', None)} / {getattr(frame, 'dtype', None)}."
        )
    if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
        raise ValueError(f"Camera {name}: missing or invalid capture timestamp.")
    if previous_timestamp is not None and timestamp <= previous_timestamp:
        raise ValueError(f"Camera {name}: frame timestamp did not advance.")
    if time.perf_counter() - timestamp > max_age:
        raise ValueError(f"Camera {name}: frame is older than {max_age:g} seconds.")


def run_preview(*, profile, args):
    import deploy.robot.communication as comms
    import zmq

    cv2 = None
    window = "YAM cameras: top | left | right (Q / Esc to quit)"
    context = zmq.Context()
    sockets = {}
    try:
        if not args.no_display:
            import cv2

            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for name in CAMERAS:
            sockets[name] = comms.create_subscriber(
                context, profile.cameras[name].socket or f"{name}_rgb", conflate=1
            )
        previous = dict.fromkeys(CAMERAS)
        count = 0
        started = last_report = time.perf_counter()
        while args.frames == 0 or count < args.frames:
            frames = {}
            for name, socket in sockets.items():
                timeout = args.startup_timeout if previous[name] is None else args.frame_timeout
                frame, extras = comms.subscribe(
                    socket, timeout_ms=int(timeout * 1000), topic_label=f"camera:{name}"
                )
                timestamp = extras.get("timestamp")
                validate_frame(name, frame, timestamp, previous[name], args.frame_timeout)
                previous[name] = timestamp
                frames[name] = frame
            count += 1
            now = time.perf_counter()
            if count == 1 or now - last_report >= 1:
                levels = ", ".join(
                    f"{name}: mean={frame.mean():.1f}" for name, frame in frames.items()
                )
                print(
                    f"frames/camera={count} preview_rate={count / max(now - started, 1e-6):.1f} Hz; {levels}",
                    flush=True,
                )
                if any(not frame.any() for frame in frames.values()):
                    print(
                        "WARNING: a camera is returning an entirely black frame; inspect its view.",
                        flush=True,
                    )
                last_report = now
            if cv2 is not None:
                panels = []
                for name, frame in frames.items():
                    panel = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (320, 240))
                    label = f"{name}: {profile.cameras[name].serial}"
                    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(
                        panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
                    panels.append(panel)
                cv2.imshow(window, np.concatenate(panels, axis=1))
                if (
                    cv2.waitKey(1) & 0xFF in (27, ord("q"))
                    or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
        print(f"Camera test finished: received {count} RGB frames per camera.", flush=True)
    finally:
        for socket in sockets.values():
            socket.close(linger=0)
        context.term()
        if cv2 is not None:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_station_arguments(parser)
    parser.add_argument(
        "--no-display", action="store_true", help="Print frame statistics without opening a window."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames per camera; 0 runs until quit.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30,
        help="Seconds to wait for each camera's first frame.",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=2,
        help="Maximum frame age and wait after startup, in seconds.",
    )
    args = parser.parse_args(argv)
    if args.frames < 0 or any(
        not np.isfinite(value) or value <= 0 for value in (args.startup_timeout, args.frame_timeout)
    ):
        parser.error("Require nonnegative --frames and finite positive timeouts.")
    profile = load_profile(args)
    for name in CAMERAS:
        camera = profile.cameras[name]
        print(
            f"{name}: {camera.camera_type}, serial={camera.serial}, {camera.width}x{camera.height} @ {camera.fps} FPS"
        )
    if args.check_only:
        return 0
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "Camera preview needs a desktop display; use --no-display --frames 90 for SSH."
        )
    from deploy.robot.launch import ProcessSpec
    from deploy.robot.specs import camera_specs

    with station_lock(args.ipc_dir) as ipc_dir:
        module = "camera_test" if __name__ == "__main__" else __name__
        specs = camera_specs(profile) + [
            ProcessSpec("camera_preview", f"{module}:run_preview", dict(profile=profile, args=args))
        ]
        return launch_nodes(specs, ipc_dir)


if __name__ == "__main__":
    raise SystemExit(main())
