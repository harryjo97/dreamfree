"""Independent wrist FK and bounded IK admission, without hardware access."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dreamfree.data import yam
from dreamfree.data.yam_eef import YamWristFK
from dreamfree.evaluation.yam_real import common
from dreamfree.evaluation.yam_real import eef_control as eef

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    Path(os.environ.get("ABC_ROOT", ROOT / "third_party/abc"))
    / "abc_sim/models/assets/i2rt_yam/yam.xml"
)
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="Set ABC_ROOT to the pinned ABC checkout for IK tests."
)
HOME = np.array([0.1, 0.6, 1.2, -0.5, 0.1, 0.2, 0.7] * 2, dtype=np.float32)


def test_standalone_fk_matches_independent_mujoco_and_training_labels():
    reference = json.loads((ROOT / "tests/fixtures/yam_wrist_fk_reference.json").read_text())
    fk = eef.WristKinematics(MODEL_PATH)
    for joints, pose in zip(reference["joints"], reference["poses"]):
        np.testing.assert_allclose(fk.forward(joints), pose, atol=2e-14)
    commands = np.concatenate(
        (reference["joints"], np.full((20, 1), 0.3), reference["joints"], np.full((20, 1), 0.7)),
        axis=1,
    )
    np.testing.assert_array_equal(
        [fk.encode_command(q) for q in commands], YamWristFK(MODEL_PATH).actions(commands)
    )


def test_ik_jacobian_matches_finite_differences():
    fk = eef.WristKinematics(MODEL_PATH)
    q = HOME[:6].astype(float)
    pose, jp, jr = fk.forward(q, jacobian=True)
    for index in range(6):
        shifted = q.copy()
        shifted[index] += 1e-6
        difference = (fk.forward(shifted) - pose) / 1e-6
        np.testing.assert_allclose(jp[:, index], difference[:3, 3], atol=3e-7)
        np.testing.assert_allclose(jr[:, index], difference[:3, :3].ravel(), atol=1e-6)


@pytest.mark.parametrize("columns", [[0] * 6, [1, 0, 0, 2, 0, 0], [np.nan] * 6])
def test_invalid_orientation_fails(columns):
    with pytest.raises(ValueError):
        eef.decode_rotation(columns)


def test_rot6d_uses_columns_and_orthonormalizes():
    rotation = eef.WristKinematics(MODEL_PATH).forward(HOME[:6])[:3, :3]
    columns = np.concatenate((rotation[:, 0] * 2, rotation[:, 1] * 3 + rotation[:, 0] * 0.4))
    np.testing.assert_allclose(eef.decode_rotation(columns), rotation, atol=2e-15)


def test_bounded_ik_tracks_nearby_poses_across_configurations():
    control = eef.EefController(MODEL_PATH, position_tolerance=1e-4, orientation_tolerance=1e-3)
    rng = np.random.default_rng(13)
    for _ in range(40):
        previous = HOME.copy()
        for start in (0, 7):
            previous[start : start + 6] = rng.uniform(
                eef.DRIVER_LIMITS[:, 0] + 0.3, eef.DRIVER_LIMITS[:, 1] - 0.3
            )
        target = previous + rng.uniform(-0.02, 0.02, 14)
        action = control.kinematics.encode_command(target)
        command, issued, report = control.command(action, previous=previous, measured=previous)
        assert (
            np.max(np.abs(command[list(common.JOINTS)] - previous[list(common.JOINTS)])) <= 0.100001
        )
        np.testing.assert_array_equal(issued, YamWristFK(MODEL_PATH).actions(command[None])[0])
        assert max(r["position_error_m"] for r in report.values()) <= 1e-4
        assert max(r["orientation_error_rad"] for r in report.values()) <= 1e-3


@pytest.mark.parametrize("failure", ["unreachable", "step", "tracking", "rotation"])
def test_ik_rejects_unexecutable_targets_without_fallback(failure):
    control = eef.EefController(MODEL_PATH)
    action = control.kinematics.encode_command(HOME)
    measured = HOME.copy()
    if failure == "unreachable":
        action[10] = 4
    elif failure == "step":
        target = HOME.copy()
        target[0] += 1
        action = control.kinematics.encode_command(target)
    elif failure == "tracking":
        measured[0] += 1
    else:
        action[13:19] = 0
    with pytest.raises(common.EefTargetRejected, match="arm:"):
        control.command(action, previous=HOME, measured=measured)


def info():
    return dict(
        protocol=common.PROTOCOL,
        **common.action_contract(yam.ABC_EEF_NAME),
        action_horizon=32,
        action_history_horizon=0,
        variant="standard",
        fps=30,
        camera_size_hw=[480, 640],
        cameras=list(common.CAMERAS),
        replan_steps=2,
        text_mode="encoder",
    )


def test_eef_handshake_requires_explicit_space_and_exact_frame():
    common.validate_server(info(), action_space="eef", replan_steps=2)
    with pytest.raises(ValueError):
        common.validate_server(info(), replan_steps=2)
    bad = info()
    bad["eef_kinematics"]["site"] = "grasp_site"
    with pytest.raises(ValueError, match="eef_kinematics"):
        common.validate_server(bad, action_space="eef", replan_steps=2)
    action = np.tile(eef.WristKinematics(MODEL_PATH).encode_command(HOME), (32, 1))
    action[:, 9], action[:, 19] = -1, 2
    prepared, clipped = common.executable_chunk(action, action_space="eef")
    assert clipped == 64
    assert np.all(prepared[:, 9] == 0) and np.all(prepared[:, 19] == 1)
    np.testing.assert_array_equal(prepared[:, 6:9], action[:, 6:9])


def test_equivalent_angles_do_not_force_a_joint_branch_jump():
    controller = eef.EefController(MODEL_PATH)
    target = HOME.astype(float)
    target[2] += 2 * np.pi
    action = controller.kinematics.encode_command(target)
    command, _, _ = controller.command(action, previous=HOME, measured=HOME)
    np.testing.assert_allclose(command, HOME, atol=1e-6)
