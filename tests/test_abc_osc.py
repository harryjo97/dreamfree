"""Offline OSC mechanics, coordinate contracts, transport ownership and fault tests."""

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreamfree.evaluation.yam_real.common import EefTargetRejected
from dreamfree.evaluation.yam_real.osc_control import (
    CONTRACT_PATH,
    ArmOsc,
    OscConfig,
    OscFault,
    WristDynamics,
    encode_pose,
    limit_target,
    rotation_exp,
    rotation_vector,
)
from dreamfree.evaluation.yam_real.osc_driver import (
    SynchronousYamDriver,
    base_gravity,
    bounded_exchange,
    bounded_motor_torque,
)
from dreamfree.evaluation.yam_real.osc_servo import OscServo

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODEL = (
    Path(os.environ.get("ABC_ROOT", ROOT / "third_party/abc"))
    / "abc_sim/models/assets/i2rt_yam/yam.xml"
)
pytestmark = pytest.mark.skipif(
    not TRAINING_MODEL.is_file(), reason="Set ABC_ROOT to the pinned ABC checkout."
)
HOME = np.array([0.1, 0.6, 1.2, -0.5, 0.1, 0.2, 0.7])


def dynamics(filename="yam_4310_linear.xml", gravity=(0.0, 0.0, -9.81)):
    source = json.loads(CONTRACT_PATH.read_text())["models"]["robot_models/yam/" + filename]
    return WristDynamics(
        source, model_file=filename, gravity=gravity, training_model=TRAINING_MODEL
    )


def target_at(dyn, q=HOME):
    pose, *_ = dyn.evaluate(q[:6], np.zeros(6))
    return encode_pose(pose[:3, 3], pose[:3, :3], q[6])


@pytest.mark.parametrize("filename", ["yam.xml", "yam_4310_linear.xml", "yam_lw_gripper.xml"])
def test_pinned_dynamics_frame_matches_training_fk(filename):
    dyn = dynamics(filename)
    reference = json.loads((ROOT / "tests/fixtures/yam_wrist_fk_reference.json").read_text())
    for q, expected in zip(reference["joints"], reference["poses"]):
        actual, *_ = dyn.evaluate(q, np.zeros(6))
        np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert dyn.model.nv == 6 and dyn.model.nu == 0


def test_jacobian_and_jdot_use_base_frame_and_tcp_site():
    dyn = dynamics()
    q, dq = HOME[:6], np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])
    pose, jac, jdot_dq, mass, bias = dyn.evaluate(q, dq)
    eps = 1e-6
    for index in range(6):
        shifted = q.copy()
        shifted[index] += eps
        moved, *_ = dyn.evaluate(shifted, dq)
        numeric = np.r_[
            (moved[:3, 3] - pose[:3, 3]) / eps,
            rotation_vector(moved[:3, :3] @ pose[:3, :3].T) / eps,
        ]
        np.testing.assert_allclose(jac[:, index], numeric, atol=3e-7)
    _, jp, *_ = dyn.evaluate(q + eps * dq, dq)
    _, jm, *_ = dyn.evaluate(q - eps * dq, dq)
    np.testing.assert_allclose(jdot_dq, (jp - jm) @ dq / (2 * eps), atol=1e-8)
    assert np.linalg.eigvalsh(mass).min() > 0
    dyn.evaluate(q, dq)
    acceleration = np.arange(6) / 10
    dyn.data.qacc[:] = acceleration
    mujoco.mj_inverse(dyn.model, dyn.data)
    np.testing.assert_allclose(dyn.data.qfrc_inverse, mass @ acceleration + bias, atol=1e-12)


def test_gravity_once_and_correct_for_mounting_and_gripper():
    dyn = dynamics()
    cfg = OscConfig(torque_rate=(1e6,) * 7)
    arm = ArmOsc(dyn, cfg, [0, -6.57])
    torque, report = arm.compute(HOME, np.zeros(7), target_at(dyn), 0.01)
    np.testing.assert_allclose(torque[:6], dyn.gravity(HOME[:6]), atol=1e-12)
    assert not np.allclose(torque[:6], dyn.gravity(HOME[:6]) * 1.3)
    tilted = base_gravity([np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    np.testing.assert_allclose(tilted, [0, -9.81, 0], atol=1e-12)
    assert not np.allclose(dynamics(gravity=tilted).gravity(HOME[:6]), torque[:6])
    assert not np.allclose(dynamics("yam.xml").gravity(HOME[:6]), torque[:6])
    assert dyn.metadata["driver_gravity_enabled"] is False


def test_cartesian_torque_produces_requested_acceleration_without_ik(monkeypatch):
    from dreamfree.evaluation.yam_real.eef_control import EefController

    monkeypatch.setattr(
        EefController, "_solve_arm", lambda *a, **k: pytest.fail("OSC must never run IK")
    )
    dyn = dynamics()
    cfg = OscConfig(torque_rate=(1e6,) * 7, inertia_damping=1e-6)
    controller = ArmOsc(dyn, cfg, [0, -6.57])
    target = target_at(dyn)
    target[:3] += [0.001, -0.002, 0.001]
    dq = np.array([0.01, -0.02, 0.01, 0.02, -0.01, 0.02, 0])
    torque, report = controller.compute(HOME, dq, target, 0.01)
    pose, jac, jdot_dq, mass, bias = dyn.evaluate(HOME[:6], dq[:6])
    acceleration = jac @ np.linalg.solve(mass, torque[:6] - bias) + jdot_dq
    expected = cfg.kp * np.r_[target[:3] - pose[:3, 3], np.zeros(3)] - cfg.kd * (jac @ dq[:6])
    np.testing.assert_allclose(acceleration, expected, atol=1e-6)
    assert not report["torque_limited"]


@pytest.mark.parametrize("angle", [0, 1e-9, 0.3, np.pi - 1e-9, np.pi])
def test_orientation_log_near_zero_and_pi(angle):
    vector = np.array([1.0, 2.0, -1.0]) / np.sqrt(6) * angle
    matrix = rotation_exp(vector)
    np.testing.assert_allclose(rotation_exp(rotation_vector(matrix)), matrix, atol=1e-9)


def test_singular_configuration_is_explicitly_damped():
    dyn = dynamics()
    q = HOME.copy()
    q[4] = np.pi / 2
    cfg = OscConfig(joint_margin=0.001, torque_rate=(1e6,) * 7)
    torque, report = ArmOsc(dyn, cfg, [0, -6.57]).compute(q, np.zeros(7), target_at(dyn, q), 0.01)
    assert np.isfinite(torque).all()
    assert report["minimum_scaled_singular_value"] < cfg.singularity_threshold
    assert report["inertia_damping"] > cfg.inertia_damping


def test_torque_and_rate_bounds_are_logged_and_persistent_saturation_faults():
    dyn = dynamics()
    cfg = OscConfig(torque_limit=(0.2,) * 7, torque_rate=(1.0,) * 7, saturation_timeout=0.025)
    controller = ArmOsc(dyn, cfg, [0, -6.57])
    for step in (1, 2):
        torque, report = controller.compute(HOME, np.zeros(7), target_at(dyn), 0.01)
        assert np.max(np.abs(torque)) <= min(0.2, step * 0.01) + 1e-15
        assert report["torque_limited"]
    with pytest.raises(OscFault, match="saturation"):
        controller.compute(HOME, np.zeros(7), target_at(dyn), 0.01)


@pytest.mark.parametrize(
    "bad", ["nan", "joint_limit", "stopping", "pose_error", "rotation", "gripper"]
)
def test_invalid_or_out_of_bounds_state_and_targets_fail(bad):
    dyn = dynamics()
    q, dq, target = HOME.copy(), np.zeros(7), target_at(dyn)
    if bad == "nan":
        q[0] = np.nan
    if bad == "joint_limit":
        q[1] = -1
    if bad == "stopping":
        q[1], dq[1] = -0.08, -1
    if bad == "pose_error":
        target[0] += 1
    if bad == "rotation":
        target[3:9] = 0
    if bad == "gripper":
        target[9] = 2
    with pytest.raises((ValueError, OscFault)):
        ArmOsc(dyn, OscConfig(), [0, -6.57]).compute(q, dq, target, 0.01)


def test_history_target_is_limited_command_not_measured_or_raw_prediction():
    dyn, cfg = dynamics(), OscConfig()
    previous = np.tile(target_at(dyn), 2)
    requested = previous.copy()
    requested[0] += 1
    requested[19] = 2
    requested[13:19] = np.r_[
        rotation_exp(np.array([0, 0, 0.5]))[:, 0], rotation_exp(np.array([0, 0, 0.5]))[:, 1]
    ]
    accepted, report = limit_target(requested, previous, 2.0, cfg)
    assert accepted[0] - previous[0] == pytest.approx(cfg.target_linear_speed / 30)
    assert accepted[19] - previous[19] == pytest.approx(cfg.gripper_speed / 30)
    assert not np.array_equal(accepted, previous) and not np.array_equal(accepted, requested)
    assert report[0]["position_scale"] < 1 and report[1]["rotation_scale"] < 1


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, value=0.01):
        self.now += value


class FakeDriver:
    def __init__(self, clock):
        self.clock, self.sent, self.disabled = clock, [], 0
        self.fail = False

    def enable(self):
        return HOME.copy(), np.zeros(7), np.full(7, self.clock())

    def exchange(self, torque):
        if self.fail:
            raise OSError("simulated CAN failure")
        self.sent.append(torque.copy())
        return self.enable()

    def disable(self):
        self.disabled += 1
        return []


def servo_fixture(*, rejection_retries=0, **config):
    clock = Clock()
    cfg = OscConfig(**config)
    drivers = [FakeDriver(clock), FakeDriver(clock)]
    servo = OscServo(
        [ArmOsc(dynamics(), cfg, [0, -6.57]) for _ in range(2)],
        drivers,
        cfg,
        clock=clock,
        rejection_retries=rejection_retries,
    )
    servo.start()
    return servo, drivers, clock


def test_servo_continues_feedback_control_during_inference_pause():
    servo, drivers, clock = servo_fixture()
    initial = servo.target.copy()
    for _ in range(100):
        clock.advance()
        snapshot = servo.tick()
    assert len(drivers[0].sent) == len(drivers[1].sent) == 100
    np.testing.assert_array_equal(servo.target, initial)
    assert snapshot["sequence"] == -1  # Servo holds are not fabricated policy-history updates.
    assert servo.timing()["achieved_hz"] == pytest.approx(100)


@pytest.mark.parametrize(
    "fault", ["feedback", "target", "deadline", "exception", "nonfinite", "misses"]
)
@pytest.mark.parametrize("rejection_retries", [0, 2])
def test_servo_faults_disable_both_arms_and_latch(fault, rejection_retries):
    servo, drivers, clock = servo_fixture(rejection_retries=rejection_retries)
    if fault == "feedback":
        servo.feedback[0][2][:] -= 1
    elif fault == "target":
        servo.target_stamp -= 4
    elif fault == "deadline":
        servo.count = 1
        clock.advance(0.05)
        servo.feedback = [d.enable() for d in drivers]
    elif fault == "exception":
        drivers[1].fail = True
    elif fault == "nonfinite":
        servo.feedback[0][0][0] = np.nan
    elif fault == "misses":
        for _ in range(3):
            clock.advance(0.015)
            servo.tick()
        clock.advance(0.015)
    with pytest.raises((OscFault, OSError, ValueError)):
        servo.tick()
    assert not servo.active and servo.fault
    assert all(d.disabled for d in drivers)
    with pytest.raises(OscFault, match="inactive"):
        servo.submit(servo.target, 1, clock(), clock())


def test_target_ack_after_servo_tick_and_sequence_replay_fails():
    servo, drivers, clock = servo_fixture()
    clock.advance()
    requested = servo.target.copy()
    requested[0] += 0.02
    servo.submit(requested, 1, clock(), clock())
    result = servo.tick()
    assert result["sequence"] == 1 and len(drivers[0].sent) == 1
    assert result["accepted_target"] != result["measured_pose"]
    assert result["accepted_target"] != requested.tolist()
    with pytest.raises(OscFault, match="sequence"):
        servo.submit(requested, 1, clock(), clock())


class FakeInterface:
    def __init__(self):
        self.commands, self.off, self.closed = [], [], False

    def _send_message_get_response(self, *args):
        return None

    def parse_recv_message(self, *args):
        return None

    def set_control(self, motor_id, motor_type, pos, vel, kp, kd, torque):
        self.commands.append((motor_id, kp, kd, torque))
        position = HOME[motor_id - 1] if motor_id < 7 else HOME[6] * -6.57
        return SimpleNamespace(id=motor_id, error_code="0x1", position=position, velocity=0.0)

    def motor_off(self, index):
        self.off.append(index)
        if index == 2:
            raise OSError("simulated disable failure")

    def close(self):
        self.closed = True


def test_transport_single_owner_zero_onboard_pd_no_gravity_or_homing():
    interface = FakeInterface()
    arm = dict(
        motor_offsets=[0] * 7,
        motor_directions=[1] * 7,
        gripper_limits=[0, -6.57],
        gripper_type="linear_4310",
    )
    clock = Clock()
    driver = SynchronousYamDriver(interface, arm, OscConfig(), clock=clock)
    q, _, _ = driver.enable()
    np.testing.assert_allclose(q, HOME)
    clock.advance(0.1)
    driver.exchange(np.arange(7) * 0.05)
    np.testing.assert_allclose(driver.last_wire_torque, np.arange(7) * 0.05, atol=0.007)
    assert all(kp == kd == 0 for _, kp, kd, _ in interface.commands)
    errors = []

    def compete():
        try:
            driver.exchange(np.zeros(7))
        except OscFault as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=compete)
    thread.start()
    thread.join()
    assert errors and "competing" in errors[0]
    errors = driver.disable()
    assert len(errors) == 1 and interface.off == list(range(1, 8)) and interface.closed
    before = len(interface.commands)
    with pytest.raises(OscFault, match="closed"):
        driver.exchange(np.ones(7))
    assert len(interface.commands) == before


def test_bounded_can_exchange_has_no_retry_and_rejects_stale_frames():
    clock = Clock()

    class Bus:
        def __init__(self):
            self.sends = 0

        def send(self, message, timeout):
            self.sends += 1

        def recv(self, timeout):
            if timeout == 0:
                return None
            clock.advance(timeout)
            return SimpleNamespace(arbitration_id=17, timestamp=time.time() - 10)

    bus = Bus()
    with pytest.raises(OscFault, match="timeout"):
        bounded_exchange(bus, object(), 17, 0.002, clock=clock)
    assert bus.sends == 1


def test_model_hash_mismatch_is_fatal():
    with pytest.raises(ValueError, match="hash mismatch"):
        WristDynamics(
            "<mujoco/>", model_file="yam.xml", gravity=[0, 0, -9.81], training_model=TRAINING_MODEL
        )


def test_wire_quantization_respects_bounds_without_extra_pd_or_gravity():
    previous = None
    for requested in [-12.0, -11.999, 12.0, 0.0, 0.001, -0.001]:
        argument, actual = bounded_motor_torque(requested, 12.0, previous, 0.05, 28.0)
        code = int((argument + 28.0) * 4095 / 56.0)
        decoded = code * 56.0 / 4095 - 28.0
        assert decoded == pytest.approx(actual, abs=1e-14)
        assert abs(actual) <= 12.0
        if previous is not None:
            assert abs(actual - previous) <= 0.05 + 1e-12
        previous = actual


def test_rejected_target_latches_and_disables_servo():
    servo, drivers, clock = servo_fixture()
    clock.advance()
    action = servo.target.copy()
    action[3:9] = 0
    with pytest.raises(OscFault, match="Degenerate"):
        servo.submit(action, 1, clock(), clock())
    assert servo.fault and not servo.active
    assert all(d.disabled for d in drivers)


def test_candidate_rejection_preserves_hold_and_budget_resets_on_acceptance():
    servo, drivers, clock = servo_fixture(rejection_retries=1, max_position_error=0.001)
    original, stamp, report = servo.target.copy(), servo.target_stamp, servo.target_report
    action = original.copy()
    action[10] += 0.01  # Right arm fails; neither arm's target may be published.
    for rejected_sequence in (1, 3):
        clock.advance()
        accepted_sequence = servo.sequence
        retained = servo.target.copy()
        previous_torques = [arm.previous_torque.copy() for arm in servo.controllers]
        with pytest.raises(EefTargetRejected, match="Cartesian bounds"):
            servo.submit(action, rejected_sequence, clock(), clock())
        assert servo.active and servo.fault is None
        assert servo.sequence == accepted_sequence and servo.consecutive_rejections == 1
        assert servo.target_stamp == stamp and servo.last_target_update == stamp
        assert servo.target_report == report
        np.testing.assert_array_equal(servo.target, retained)
        for arm, torque in zip(servo.controllers, previous_torques):
            np.testing.assert_array_equal(arm.previous_torque, torque)
        snapshot = servo.tick()
        assert snapshot["sequence"] == accepted_sequence
        assert not any(d.disabled for d in drivers)
        clock.advance()
        servo.submit(original, rejected_sequence + 1, clock(), clock())
        assert servo.consecutive_rejections == 0
        stamp, report = servo.target_stamp, servo.target_report
        servo.tick()


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_servo_enforces_its_own_rejection_budget(retries):
    servo, drivers, clock = servo_fixture(rejection_retries=retries)
    action = servo.target.copy()
    action[3:9] = 0
    for sequence in range(1, retries + 1):
        clock.advance()
        with pytest.raises(EefTargetRejected):
            servo.submit(action, sequence, clock(), clock())
        servo.tick()
    clock.advance()
    with pytest.raises(OscFault, match="retry limit exhausted"):
        servo.submit(action, retries + 1, clock(), clock())
    assert "retry limit exhausted" in servo.fault
    assert servo.sequence == -1 and not servo.active and all(d.disabled for d in drivers)


def test_rejected_sequence_cannot_be_replayed():
    servo, drivers, clock = servo_fixture(rejection_retries=2)
    clock.advance()
    action = servo.target.copy()
    action[3:9] = 0
    with pytest.raises(EefTargetRejected):
        servo.submit(action, 1, clock(), clock())
    with pytest.raises(OscFault, match="sequence"):
        servo.submit(servo.target, 1, clock(), clock())
    assert all(d.disabled for d in drivers)


def test_rejections_do_not_extend_target_expiry():
    servo, drivers, clock = servo_fixture(rejection_retries=10, target_timeout=0.025)
    stamp = servo.target_stamp
    action = servo.target.copy()
    action[3:9] = 0
    for sequence in (1, 2):
        clock.advance()
        with pytest.raises(EefTargetRejected):
            servo.submit(action, sequence, clock(), clock())
        servo.tick()
        assert servo.target_stamp == stamp
    clock.advance()
    with pytest.raises(OscFault, match="expired"):
        servo.submit(servo.target, 3, clock(), clock())
    assert all(d.disabled for d in drivers)


@pytest.mark.parametrize("fault", ["active_target", "joint", "nonfinite"])
def test_new_target_cannot_recover_controller_fault(fault):
    servo, drivers, clock = servo_fixture(rejection_retries=2)
    candidate = servo.target.copy()
    if fault == "active_target":
        servo.target[0] += servo.config.max_position_error * 2
    elif fault == "joint":
        servo.feedback[0][0][0] = servo.config.joint_upper[0] + 0.1
    else:
        candidate[0] = np.nan
    clock.advance()
    with pytest.raises((OscFault, ValueError)) as error:
        servo.submit(candidate, 1, clock(), clock())
    assert not isinstance(error.value, EefTargetRejected)
    assert not servo.active and all(d.disabled for d in drivers)


def test_capture_bounds_endpoint_quantization_without_changing_measured_aperture():
    servo, _, clock = servo_fixture()
    servo.feedback[0][0][6] = 1.00002
    servo.feedback[1][0][6] = -0.00002
    servo.reset_target(clock())
    np.testing.assert_array_equal(servo.target[[9, 19]], [1, 0])
    np.testing.assert_array_equal(servo.measured_action()[[9, 19]], [1.00002, -0.00002])
    assert all(report["gripper_limited"] for report in servo.target_report)
    assert servo.sequence == -1


def test_mujoco_tracking_settles_without_ik():
    from dreamfree.evaluation.yam_real.validate_osc import run_tracking

    report = run_tracking(seconds=2.0)
    assert max(report["final_position_error_m"]) < 0.0005
    assert max(report["final_orientation_error_rad"]) < 0.005
    assert np.all(np.array(report["maximum_abs_torque_nm"]) <= OscConfig().torque_limit)


def test_config_refuses_invalid_limits_and_unknown_settings():
    with pytest.raises(ValueError, match="positive"):
        OscConfig(kp=[0] * 6)
    with pytest.raises(ValueError, match="pinned driver"):
        OscConfig(joint_lower=[-10] * 6)
    with pytest.raises(TypeError):
        OscConfig(unknown_gain=1)
