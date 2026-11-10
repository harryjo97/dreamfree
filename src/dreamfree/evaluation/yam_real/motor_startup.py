"""Bounded CAN retries during pinned ABC/I2RT follower initialization only."""

import hashlib
import importlib.util
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

I2RT_REVISION = "852f6fff33970fe9b882c53e6b28c66172c97291"
SOURCE_HASHES = {
    "motor_drivers/can_interface.py": "8188273f88591df4d6838fe6b1e4b50e7bceea3bf13cb08096d5ac2e50336ff6",
    "motor_drivers/dm_driver.py": "ae100ac40fc20712241cfdb9ec333fe36b6f5b8ba5735edfed8fd96c7a9f2e0d",
    "robots/get_robot.py": "c2f3c76caa0d5fb6cf85cc2e1884a217c382b37608315514e2949c9ac672b325",
    "robots/motor_chain_robot.py": "be610bd3a4da816fbb03293e513cb3d45476d5950b21a197b55c07b705d5aa16",
}
ATTEMPT_TIMEOUT = 0.2
MAX_EXTRA_RETRIES = 10


def validate_driver_sources():
    """Check the transport/factory contract without importing or opening hardware."""
    spec = importlib.util.find_spec("i2rt")
    if spec is None or spec.origin is None:
        raise RuntimeError("--motor-startup-retries requires ABC's pinned I2RT installation.")
    root = Path(spec.origin).parent
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Motor startup retry driver mismatch: {path}; expected I2RT {I2RT_REVISION}."
            )
    return dict(
        driver_revision=I2RT_REVISION,
        source_sha256=SOURCE_HASHES.copy(),
        scope="get_yam_robot_initialization_and_calibration",
        attempt_timeout_s=ATTEMPT_TIMEOUT,
    )


def exchange(interface, frame_id, motor_id, data, max_retry, expected_id, extra_retries, active):
    """Retry identical requests and wait past unrelated replies within each deadline."""
    import can

    if interface.use_buffered_reader:
        raise ValueError("Motor startup retries require the pinned unbuffered SocketCAN transport.")
    expected_id = (
        interface.receive_mode.get_receive_id(motor_id) if expected_id is None else expected_id
    )
    message = can.Message(arbitration_id=frame_id, data=data, is_extended_id=False)
    attempts = max_retry + extra_retries
    started = time.monotonic()
    # A delayed reply to an identical retry is usable; frames predating the
    # first send are not. CAN provides no request sequence number.
    sent_after = time.time()
    ignored = 0
    reason = "no matching response"
    for attempt in range(attempts):
        # Initialization may finish while the driver's background loop is in
        # this call. Do not grant extra attempts once the factory has returned.
        if attempt >= max_retry and not active.is_set():
            break
        deadline = time.monotonic() + ATTEMPT_TIMEOUT
        try:
            interface.bus.send(message, timeout=ATTEMPT_TIMEOUT)
            while (remaining := deadline - time.monotonic()) > 0:
                response = interface.bus.recv(timeout=remaining)
                if response is None:
                    break
                if (
                    response.arbitration_id != expected_id
                    or response.is_extended_id
                    or response.is_remote_frame
                    or response.is_error_frame
                    or response.timestamp < sent_after
                ):
                    ignored += 1
                    continue
                if time.monotonic() > deadline:
                    reason = "matching response arrived after the attempt deadline"
                    break
                if len(response.data) != 8:
                    raise ValueError(
                        f"Motor {motor_id} returned a malformed startup response on {interface.bus.channel_info}."
                    )
                if attempt or ignored:
                    print(
                        f"[motor-startup-retry] motor={motor_id} channel={interface.bus.channel_info}: "
                        f"recovered on attempt {attempt + 1}/{attempts}; ignored_frames={ignored}",
                        file=sys.stderr,
                        flush=True,
                    )
                # Motor status is parsed by the original driver; hardware
                # fault responses are not converted into communication retries.
                return response
        except can.CanError as exc:
            reason = str(exc)
        remaining_attempts = attempts if active.is_set() else max_retry
        if attempt + 1 < remaining_attempts:
            print(
                f"[motor-startup-retry] motor={motor_id} channel={interface.bus.channel_info}: "
                f"attempt {attempt + 1}/{attempts} failed ({reason}); retrying the same request",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.001)
    raise AssertionError(
        f"fail to communicate with the motor {frame_id} on {interface.name} "
        f"at can channel {interface.bus.channel_info}; startup request exhausted "
        f"after {time.monotonic() - started:.3f}s: {reason}; ignored_frames={ignored}"
    )


@contextmanager
def startup_retries(extra_retries):
    """Install a process-local adapter around robot creation, never restart it."""
    if type(extra_retries) is not int or not 0 <= extra_retries <= MAX_EXTRA_RETRIES:
        raise ValueError(f"Motor startup retries must be an integer in 0..{MAX_EXTRA_RETRIES}.")
    if extra_retries == 0:
        yield
        return
    validate_driver_sources()
    import i2rt.motor_drivers.dm_driver as driver
    import i2rt.robots.get_robot as factory

    original_interface = driver.DMSingleMotorCanInterface
    original_factory = factory.get_yam_robot
    active = threading.Event()

    class StartupInterface(original_interface):
        def _send_message_get_response(self, id, motor_id, data, max_retry=5, expected_id=None):
            if not active.is_set():
                return super()._send_message_get_response(
                    id, motor_id, data, max_retry, expected_id
                )
            return exchange(self, id, motor_id, data, max_retry, expected_id, extra_retries, active)

    def open_robot(*args, **kwargs):
        active.set()
        print(
            f"[motor-startup-retry] enabled: {extra_retries} extra attempts per CAN request "
            "during initialization/calibration",
            file=sys.stderr,
            flush=True,
        )
        try:
            return original_factory(*args, **kwargs)
        finally:
            active.clear()

    driver.DMSingleMotorCanInterface = StartupInterface
    factory.get_yam_robot = open_robot
    try:
        yield
    finally:
        active.clear()
        factory.get_yam_robot = original_factory
        driver.DMSingleMotorCanInterface = original_interface
