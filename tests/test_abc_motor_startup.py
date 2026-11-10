"""Motor startup transport behavior with fake CAN; no hardware SDK is needed."""

import hashlib
import sys
import threading
from collections import deque
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from dreamfree.evaluation.yam_real import client as robot
from dreamfree.evaluation.yam_real import motor_startup as startup


class CanError(Exception):
    pass


@pytest.fixture
def transport(monkeypatch):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def time(self):
            return 100.0 + self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(startup, "time", clock)
    monkeypatch.setitem(
        sys.modules, "can", SimpleNamespace(CanError=CanError, Message=SimpleNamespace)
    )

    class Bus:
        channel_info = "socketcan channel 'can_follower_l'"

        def __init__(self, responses=(), send_errors=()):
            self.responses = deque(responses)
            self.send_errors = deque(send_errors)
            self.sent = []

        def send(self, message, timeout):
            assert 0 < timeout <= startup.ATTEMPT_TIMEOUT
            self.sent.append(message)
            if self.send_errors:
                error = self.send_errors.popleft()
                if error is not None:
                    raise error

        def recv(self, timeout):
            response = self.responses.popleft() if self.responses else None
            clock.sleep(timeout if response is None else min(0.001, timeout))
            if isinstance(response, Exception):
                raise response
            return response

    def interface(responses=(), send_errors=()):
        return SimpleNamespace(
            bus=Bus(responses, send_errors),
            name="yam_real",
            use_buffered_reader=False,
            receive_mode=SimpleNamespace(get_receive_id=lambda motor: motor + 16),
        )

    active = threading.Event()
    active.set()
    return interface, active, clock


def frame(id=20, *, timestamp=100.0, data=b"\x14" + bytes(7), **kwargs):
    return SimpleNamespace(
        arbitration_id=id,
        timestamp=timestamp,
        data=data,
        is_extended_id=False,
        is_remote_frame=False,
        is_error_frame=False,
        **kwargs,
    )


def request(interface, active, *, retries=2, expected_id=None):
    return startup.exchange(interface, 4, 4, bytes(8), 2, expected_id, retries, active)


def test_waits_past_other_motor_and_old_frames_without_resending(transport, capsys):
    make_interface, active, _ = transport
    expected = frame()
    interface = make_interface([frame(21), frame(timestamp=99.0), expected])
    assert request(interface, active) is expected
    assert len(interface.bus.sent) == 1
    assert "ignored_frames=2" in capsys.readouterr().err


def test_extra_attempt_recovers_same_request(transport, capsys):
    make_interface, active, _ = transport
    expected = frame(timestamp=101)
    interface = make_interface([None, None, expected])
    assert request(interface, active) is expected
    assert len(interface.bus.sent) == 3
    assert all(message is interface.bus.sent[0] for message in interface.bus.sent)
    output = capsys.readouterr().err
    assert "attempt 2/4 failed" in output and "recovered on attempt 3/4" in output


@pytest.mark.parametrize("send_error", [True, False])
def test_can_errors_retry_but_still_require_a_response(transport, send_error):
    make_interface, active, _ = transport
    expected = frame(timestamp=101)
    interface = make_interface(
        [expected] if send_error else [CanError("receive failed"), expected],
        [CanError("send failed")] if send_error else [],
    )
    assert request(interface, active) is expected
    assert len(interface.bus.sent) == 2


def test_missing_motor_exhausts_finite_budget(transport, capsys):
    make_interface, active, clock = transport
    interface = make_interface()
    with pytest.raises(AssertionError, match="motor 4.*can_follower_l.*startup request exhausted"):
        request(interface, active)
    assert len(interface.bus.sent) == 4
    assert clock.now == pytest.approx(4 * 0.2 + 3 * 0.001)
    assert "recovered" not in capsys.readouterr().err


def test_continuous_wrong_ids_cannot_extend_attempt_deadlines(transport):
    make_interface, active, clock = transport
    interface = make_interface([frame(21, timestamp=101) for _ in range(2000)])
    with pytest.raises(AssertionError, match="ignored_frames="):
        request(interface, active)
    assert len(interface.bus.sent) == 4
    assert clock.now <= 0.804


def test_explicit_expected_id_and_fault_payload_are_preserved(transport):
    make_interface, active, _ = transport
    # A motor fault is a valid reply, parsed by I2RT after exchange returns.
    expected = frame(42, data=b"\x84" + bytes(7))
    interface = make_interface([expected])
    assert request(interface, active, expected_id=42) is expected
    assert len(interface.bus.sent) == 1


def test_malformed_response_does_not_retry(transport):
    make_interface, active, _ = transport
    interface = make_interface([frame(data=b"bad")])
    with pytest.raises(ValueError, match="malformed"):
        request(interface, active)
    assert len(interface.bus.sent) == 1


def test_extra_attempts_stop_when_initialization_finishes(transport):
    make_interface, active, _ = transport
    interface = make_interface()
    original_recv = interface.bus.recv

    def recv(timeout):
        active.clear()
        return original_recv(timeout)

    interface.bus.recv = recv
    with pytest.raises(AssertionError):
        request(interface, active)
    assert len(interface.bus.sent) == 2


@pytest.mark.parametrize("failure", [None, RuntimeError("motor fault"), KeyboardInterrupt()])
def test_factory_runs_once_and_runtime_uses_original_transport(monkeypatch, failure):
    driver = ModuleType("i2rt.motor_drivers.dm_driver")
    factory = ModuleType("i2rt.robots.get_robot")
    events = []

    class OriginalInterface:
        def _send_message_get_response(self, *args):
            events.append("runtime")
            return "original response"

    def original_factory():
        events.append("factory")
        interface = driver.DMSingleMotorCanInterface()
        assert interface._send_message_get_response(4, 4, bytes(8)) == "startup response"
        if failure:
            raise failure
        return interface

    driver.DMSingleMotorCanInterface = OriginalInterface
    factory.get_yam_robot = original_factory
    packages = {name: ModuleType(name) for name in ("i2rt", "i2rt.motor_drivers", "i2rt.robots")}
    packages["i2rt"].motor_drivers = packages["i2rt.motor_drivers"]
    packages["i2rt"].robots = packages["i2rt.robots"]
    packages["i2rt.motor_drivers"].dm_driver = driver
    packages["i2rt.robots"].get_robot = factory
    for name, module in {**packages, driver.__name__: driver, factory.__name__: factory}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(startup, "validate_driver_sources", lambda: {})

    def exchange(*args):
        assert args[-1].is_set()
        events.append("startup")
        return "startup response"

    monkeypatch.setattr(startup, "exchange", exchange)
    try:
        with startup.startup_retries(3):
            interface = factory.get_yam_robot()
            assert interface._send_message_get_response(4, 4, bytes(8)) == "original response"
    except BaseException as exc:
        assert exc is failure
    else:
        assert failure is None
    assert events.count("factory") == 1 and events.count("startup") == 1
    assert driver.DMSingleMotorCanInterface is OriginalInterface
    assert factory.get_yam_robot is original_factory


def test_source_check_rejects_modified_driver_before_import(tmp_path, monkeypatch):
    monkeypatch.setattr(
        startup.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py")),
    )
    content = b"pinned fixture"
    monkeypatch.setattr(
        startup, "SOURCE_HASHES", {"driver.py": hashlib.sha256(content).hexdigest()}
    )
    (tmp_path / "driver.py").write_bytes(content)
    assert startup.validate_driver_sources()["driver_revision"] == startup.I2RT_REVISION
    (tmp_path / "driver.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="driver mismatch"):
        startup.validate_driver_sources()


def test_cli_rejects_unused_or_unbounded_retries():
    base = ["--instruction", "task", "--check-only"]
    assert robot.parse_args(base).motor_startup_retries == 5
    assert robot.parse_args(base + ["--action-space", "eef"]).motor_startup_retries == 5
    assert robot.parse_args(base + ["--debug"]).motor_startup_retries == 0
    assert (
        robot.parse_args(
            base
            + ["--action-space", "eef", "--eef-controller", "osc", "--osc-config", "station.json"]
        ).motor_startup_retries
        == 0
    )
    assert robot.parse_args(base + ["--motor-startup-retries", "0"]).motor_startup_retries == 0
    assert robot.parse_args(base + ["--motor-startup-retries", "3"]).motor_startup_retries == 3
    for extra in (
        ["--motor-startup-retries", "-1"],
        ["--motor-startup-retries", "11"],
        ["--motor-startup-retries", "1", "--debug"],
        ["--motor-startup-retries", "1", "--action-space", "eef", "--eef-controller", "osc"],
    ):
        with pytest.raises(SystemExit):
            robot.parse_args(base + extra)


@pytest.mark.parametrize(
    "target", ["deploy.robot.followers.yam_follower", "camera", "osc_servo:run"]
)
def test_child_adapter_only_wraps_joint_followers(monkeypatch, target):
    events = []
    for name in ("deploy", "deploy.robot", "deploy.robot.communication"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "dreamfree.evaluation.yam_real.motor_startup", startup)

    @contextmanager
    def retries(count):
        assert count == 5
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(startup, "startup_retries", retries)
    import_module = robot.importlib.import_module
    module_name = target.split(":")[0]
    monkeypatch.setattr(
        robot.importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            SimpleNamespace(run=lambda: events.append("run"))
            if name == module_name
            else import_module(name, *args, **kwargs)
        ),
    )
    robot.run_node(target=target, kwargs={}, ipc_dir="station_ipc")
    assert events == (["enter", "run", "exit"] if "yam_follower" in target else ["run"])
