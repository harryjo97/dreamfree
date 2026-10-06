from __future__ import annotations

import ast
import functools
import importlib
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TASK_MAX_STEPS: dict[str, int] = {
    "ArrangeVegetables": 1200,
    "MicrowaveThawing": 3200,
    "PrepareCoffee": 1500,
    "PreSoakPan": 2000,
    "RestockPantry": 2400,
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}
ROBOCASA_EVAL_TASK_ALIASES: dict[str, str] = {
    "PrepareCoffee038": "PrepareCoffee",
    "PrepareCoffeeMixed": "PrepareCoffee",
}
PREPARE_COFFEE_MIXED_ASSETS: tuple[str, ...] = (
    "CoffeeMachine038",
    "CoffeeMachine044",
    "CoffeeMachine048",
    "CoffeeMachine049",
    "CoffeeMachine057",
    "CoffeeMachine059",
    "CoffeeMachine060",
    "CoffeeMachine063",
    "CoffeeMachine070",
    "CoffeeMachine073",
    "CoffeeMachine074",
    "CoffeeMachine075",
    "CoffeeMachine076",
    "CoffeeMachine077",
    "CoffeeMachine080",
    "CoffeeMachine081",
    "CoffeeMachine082",
    "CoffeeMachine083",
    "CoffeeMachine088",
    "CoffeeMachine089",
    "CoffeeMachine090",
    "CoffeeMachine091",
    "CoffeeMachine092",
    "CoffeeMachine093",
    "CoffeeMachine095",
    "CoffeeMachine096",
    "CoffeeMachine097",
    "CoffeeMachine098",
    "CoffeeMachine099",
    "CoffeeMachine100",
    "CoffeeMachine101",
    "CoffeeMachine103",
)
PREPARE_COFFEE_VARIANT_SCENE: tuple[tuple[int, int], ...] = ((1, 1),)
ENV_TO_DREAMFREE_IMAGE_KEYS: tuple[tuple[str, str], ...] = (
    ("robot0_agentview_left_image", "agentview_left"),
    ("robot0_agentview_right_image", "agentview_right"),
    ("robot0_eye_in_hand_image", "wrist"),
)
INITIAL_STATE_VALIDATION_TASKS: tuple[str, ...] = ("PreSoakPan",)
INITIAL_STATE_MAX_Z_ERROR = 0.05
INITIAL_STATE_MAX_LINEAR_SPEED = 0.05
_ROBOSUITE_CONTEXT_GUARD_MARKER = "_dreamfree_restores_offscreen_gl_context"


def resolve_path(path_value: str | os.PathLike[str], *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def ensure_robocasa_import(robocasa_root: str | os.PathLike[str], *, project_root: Path) -> Path:
    root = resolve_path(robocasa_root, base=project_root)
    if not root.exists():
        raise FileNotFoundError(f"RoboCasa root not found: {root}")
    package_dir = root / "robocasa"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"RoboCasa package directory not found: {package_dir}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    robocasa_module = importlib.import_module("robocasa")
    module_file = getattr(robocasa_module, "__file__", None)
    if module_file is None:
        raise RuntimeError(
            f"Imported robocasa package has no __file__; expected package under {package_dir}."
        )
    imported_package_dir = Path(module_file).resolve().parent
    if imported_package_dir != package_dir:
        raise RuntimeError(
            f"Imported robocasa package does not match EVALUATION.robocasa_root: expected={package_dir} imported={imported_package_dir}."
        )
    install_robosuite_offscreen_context_guard()
    return root


def install_robosuite_offscreen_context_guard() -> None:
    """Restore robosuite's GL context before every offscreen render."""
    try:
        binding_utils = importlib.import_module("robosuite.utils.binding_utils")
        render_context_cls = binding_utils.MjRenderContext
        original_render = render_context_cls.render
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "RoboCasa evaluation requires robosuite.utils.binding_utils.MjRenderContext.render."
        ) from exc
    if bool(getattr(original_render, _ROBOSUITE_CONTEXT_GUARD_MARKER, False)):
        return

    @functools.wraps(original_render)
    def render_with_current_context(self, *args, **kwargs):
        gl_context = getattr(self, "gl_ctx", None)
        make_current = getattr(gl_context, "make_current", None)
        if not callable(make_current):
            raise RuntimeError(
                "RoboCasa offscreen rendering requires MjRenderContext.gl_ctx.make_current()."
            )
        make_current()
        return original_render(self, *args, **kwargs)

    setattr(render_with_current_context, _ROBOSUITE_CONTEXT_GUARD_MARKER, True)
    render_context_cls.render = render_with_current_context


def set_seed_everywhere(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_controller_configs(
    path_value: str | os.PathLike[str] | None, controllers: str, *, project_root: Path
):
    if path_value is not None and str(path_value).strip().lower() not in {"", "none", "null"}:
        path = resolve_path(path_value, base=project_root)
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
    from robosuite import load_controller_config

    return load_controller_config(default_controller=str(controllers))


def get_all_task_names() -> set[str]:
    from robocasa.utils.dataset_registry import (
        MULTI_STAGE_TASK_DATASETS,
        SINGLE_STAGE_TASK_DATASETS,
    )

    return (
        set(SINGLE_STAGE_TASK_DATASETS)
        | set(MULTI_STAGE_TASK_DATASETS)
        | set(ROBOCASA_EVAL_TASK_ALIASES)
    )


def get_base_task_name(task_name: str) -> str:
    task_name = str(task_name)
    return ROBOCASA_EVAL_TASK_ALIASES.get(task_name, task_name)


def get_prepare_coffee_machine_asset(task_name: str, episode_idx: int) -> str | None:
    task_name = str(task_name)
    episode_idx = int(episode_idx)
    if episode_idx < 0:
        raise ValueError(f"episode_idx must be non-negative, got {episode_idx}.")
    if task_name == "PrepareCoffee038":
        return "CoffeeMachine038"
    if task_name == "PrepareCoffeeMixed":
        return PREPARE_COFFEE_MIXED_ASSETS[episode_idx % len(PREPARE_COFFEE_MIXED_ASSETS)]
    return None


def validate_task_name(task_name: str) -> None:
    if task_name in ROBOCASA_EVAL_TASK_ALIASES:
        return
    all_tasks = get_all_task_names()
    if task_name not in all_tasks:
        preview = ", ".join(sorted(all_tasks)[:20])
        raise ValueError(
            f"Unknown RoboCasa task_name={task_name!r}. Known tasks include: {preview} ... ({len(all_tasks)} total)"
        )


def parse_layout_and_style_ids(value: Any) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        value = ast.literal_eval(text)
    parsed = tuple(tuple(int(x) for x in item) for item in value)
    for item in parsed:
        if len(item) != 2:
            raise ValueError(f"Each layout/style id must have length 2, got: {item}")
    return parsed


def select_layout_and_style_ids(
    value: Any, episode_idx: int | None
) -> tuple[tuple[int, int], ...] | None:
    all_ids = parse_layout_and_style_ids(value)
    if all_ids is None or episode_idx is None:
        return all_ids
    if len(all_ids) == 0:
        return all_ids
    scene_index = int(episode_idx) // 10 % len(all_ids)
    return (all_ids[scene_index],)


def create_robocasa_env(
    eval_cfg: Any, *, project_root: Path, seed: int | None, episode_idx: int | None
):
    import robosuite

    controller_configs = load_controller_configs(
        getattr(eval_cfg, "controller_configs_path", None),
        getattr(eval_cfg, "controllers", "OSC_POSE"),
        project_root=project_root,
    )
    requested_task_name = str(eval_cfg.task_name)
    base_task_name = get_base_task_name(requested_task_name)
    is_prepare_coffee_variant = requested_task_name in ROBOCASA_EVAL_TASK_ALIASES
    if is_prepare_coffee_variant:
        layout_and_style_ids = PREPARE_COFFEE_VARIANT_SCENE
    else:
        layout_and_style_ids = select_layout_and_style_ids(
            getattr(eval_cfg, "layout_and_style_ids", None), episode_idx
        )
    env_kwargs = dict(
        env_name=base_task_name,
        robots=str(getattr(eval_cfg, "robots", "PandaMobile")),
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=int(getattr(eval_cfg, "env_img_res", 224)),
        camera_heights=int(getattr(eval_cfg, "env_img_res", 224)),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=str(getattr(eval_cfg, "obj_instance_split", "B")),
        generative_textures=None,
        randomize_cameras=bool(getattr(eval_cfg, "randomize_cameras", False)),
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    if is_prepare_coffee_variant:
        if episode_idx is None:
            raise ValueError(
                f"RoboCasa task {requested_task_name} requires an episode index for fixture selection."
            )
        coffee_machine_asset = get_prepare_coffee_machine_asset(requested_task_name, episode_idx)
        if coffee_machine_asset is None:
            raise RuntimeError(
                f"RoboCasa task {requested_task_name} did not resolve a coffee-machine asset."
            )
        env_kwargs["fixture_style_overrides"] = {"coffee_machine": coffee_machine_asset}
    return (robosuite.make(**env_kwargs), env_kwargs)


def _copy_rgb(image: Any, *, flip: bool) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape={array.shape}")
    if flip:
        array = np.flipud(array)
    return np.ascontiguousarray(array, dtype=np.uint8)


def prepare_robocasa_observation(
    obs: Mapping[str, Any], *, flip_images: bool = True
) -> dict[str, np.ndarray]:
    missing = [env_key for (env_key, _) in ENV_TO_DREAMFREE_IMAGE_KEYS if env_key not in obs]
    if missing:
        raise KeyError(f"RoboCasa observation is missing image keys: {missing}")
    return {
        dreamfree_key: _copy_rgb(obs[env_key], flip=flip_images)
        for (env_key, dreamfree_key) in ENV_TO_DREAMFREE_IMAGE_KEYS
    }


def extract_robocasa_state(obs: Mapping[str, Any]) -> np.ndarray:
    keys = ("robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(f"RoboCasa observation is missing state keys: {missing}")
    return np.concatenate(
        [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys]
    ).astype(np.float32)


def get_dummy_action(env) -> np.ndarray:
    (low, _high) = env.action_spec
    return np.zeros_like(low, dtype=np.float32)


def validate_robocasa_initial_state(
    env,
    task_name: str,
    *,
    max_z_error: float = INITIAL_STATE_MAX_Z_ERROR,
    max_linear_speed: float = INITIAL_STATE_MAX_LINEAR_SPEED,
) -> dict[str, Any]:
    """Validate task-specific initial-state invariants after reset warmup."""
    task_name = str(task_name)
    if task_name not in INITIAL_STATE_VALIDATION_TASKS:
        return {"valid": True, "applied": False, "validator": None, "objects": []}
    if task_name != "PreSoakPan":
        raise ValueError(
            f"No RoboCasa initial-state validator is implemented for task={task_name!r}."
        )
    if max_z_error < 0:
        raise ValueError(f"max_z_error must be non-negative, got {max_z_error}.")
    if max_linear_speed < 0:
        raise ValueError(f"max_linear_speed must be non-negative, got {max_linear_speed}.")
    counter = getattr(env, "counter", None)
    if counter is None:
        raise RuntimeError("PreSoakPan initial-state validation requires env.counter.")
    diagnostics = []
    for obj_name in ("obj1", "obj2"):
        if obj_name not in env.objects:
            raise RuntimeError(
                f"PreSoakPan initial-state validation is missing env.objects[{obj_name!r}]."
            )
        if obj_name not in env.object_placements:
            raise RuntimeError(
                f"PreSoakPan initial-state validation is missing env.object_placements[{obj_name!r}]."
            )
        if obj_name not in env.obj_body_id:
            raise RuntimeError(
                f"PreSoakPan initial-state validation is missing env.obj_body_id[{obj_name!r}]."
            )
        obj = env.objects[obj_name]
        if len(obj.joints) != 1:
            raise RuntimeError(
                f"PreSoakPan initial-state validation expected one free joint for {obj_name}, got joints={obj.joints}."
            )
        sampled_pos = np.asarray(env.object_placements[obj_name][0], dtype=np.float64)
        current_pos = np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=np.float64
        )
        if sampled_pos.shape != (3,) or current_pos.shape != (3,):
            raise RuntimeError(
                f"PreSoakPan initial-state validation expected 3D positions for {obj_name}, got sampled={sampled_pos.shape} current={current_pos.shape}."
            )
        joint_id = int(env.sim.model.joint_name2id(obj.joints[0]))
        dof_addr = int(env.sim.model.jnt_dofadr[joint_id])
        linear_velocity = np.asarray(env.sim.data.qvel[dof_addr : dof_addr + 3], dtype=np.float64)
        if linear_velocity.shape != (3,):
            raise RuntimeError(
                f"PreSoakPan initial-state validation expected 3D linear velocity for {obj_name}, got shape={linear_velocity.shape}."
            )
        counter_contact = bool(env.check_contact(obj, counter))
        z_error = abs(float(current_pos[2] - sampled_pos[2]))
        linear_speed = float(np.linalg.norm(linear_velocity))
        obj_valid = (
            counter_contact
            and z_error <= float(max_z_error)
            and (linear_speed <= float(max_linear_speed))
        )
        diagnostics.append(
            {
                "name": obj_name,
                "valid": bool(obj_valid),
                "counter_contact": counter_contact,
                "sampled_z": float(sampled_pos[2]),
                "current_z": float(current_pos[2]),
                "z_error": z_error,
                "linear_speed": linear_speed,
            }
        )
    return {
        "valid": all(item["valid"] for item in diagnostics),
        "applied": True,
        "validator": "presoakpan_counter_support",
        "objects": diagnostics,
    }


def _format_initial_state_diagnostics(validation: Mapping[str, Any]) -> str:
    object_diagnostics = validation.get("objects", [])
    if not object_diagnostics:
        return f"validator={validation.get('validator')} valid={validation.get('valid')}"
    return "; ".join(
        f"{item['name']}(valid={item['valid']}, contact={item['counter_contact']}, sampled_z={item['sampled_z']:.5f}, current_z={item['current_z']:.5f}, z_error={item['z_error']:.5f}, linear_speed={item['linear_speed']:.5f})"
        for item in object_diagnostics
    )


def reset_and_warm_up_robocasa_env(
    env,
    *,
    task_name: str,
    num_steps_wait: int,
    max_reset_attempts: int,
    episode_idx: int,
    seed: int | None,
    validation_enabled: bool,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Reset and warm up an environment until its registered validator passes."""
    num_steps_wait = int(num_steps_wait)
    max_reset_attempts = int(max_reset_attempts)
    if num_steps_wait < 0:
        raise ValueError(f"num_steps_wait must be non-negative, got {num_steps_wait}.")
    if max_reset_attempts < 1:
        raise ValueError(f"max_reset_attempts must be at least 1, got {max_reset_attempts}.")
    task_name = str(task_name)
    validation_applies = bool(validation_enabled) and task_name in INITIAL_STATE_VALIDATION_TASKS
    attempt_limit = max_reset_attempts if validation_applies else 1
    last_validation: dict[str, Any] | None = None
    for attempt in range(1, attempt_limit + 1):
        obs = env.reset()
        if obs is None:
            raise RuntimeError(
                f"RoboCasa env.reset() returned no observation for task={task_name} episode_idx={episode_idx} seed={seed} attempt={attempt}."
            )
        for _ in range(num_steps_wait):
            (obs, _reward, _done, _info) = env.step(get_dummy_action(env))
        if obs is None:
            raise RuntimeError(
                f"Failed to collect a warmed RoboCasa observation for task={task_name} episode_idx={episode_idx} seed={seed} attempt={attempt}."
            )
        if validation_applies:
            last_validation = validate_robocasa_initial_state(env, task_name)
        else:
            last_validation = {"valid": True, "applied": False, "validator": None, "objects": []}
        if bool(last_validation["valid"]):
            if attempt > 1:
                logging.info(
                    "Accepted RoboCasa reset after retry | task=%s episode_idx=%d seed=%s attempts=%d",
                    task_name,
                    episode_idx,
                    seed,
                    attempt,
                )
            return (obs, attempt, last_validation)
        logging.warning(
            "Rejected invalid RoboCasa reset | task=%s episode_idx=%d seed=%s attempt=%d/%d | %s",
            task_name,
            episode_idx,
            seed,
            attempt,
            attempt_limit,
            _format_initial_state_diagnostics(last_validation),
        )
    assert last_validation is not None
    raise RuntimeError(
        f"RoboCasa initial-state validation failed after {attempt_limit} reset attempts: task={task_name} episode_idx={episode_idx} seed={seed} | {_format_initial_state_diagnostics(last_validation)}"
    )


def pad_action_for_env(action: np.ndarray, env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(getattr(env, "action_dim", env.action_spec[0].shape[0]))
    if action.shape[0] == env_action_dim:
        return action
    if action.shape[0] == 7 and env_action_dim == 12:
        mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        return np.concatenate([action, mobile_base_action], axis=0)
    raise ValueError(
        f"Cannot adapt DreamFree action dim {action.shape[0]} to env action dim {env_action_dim}."
    )


def record_issued_action_if_enabled(policy: Any, action: np.ndarray) -> None:
    if getattr(policy, "action_history_horizon", None) is None:
        return
    record_fn = getattr(policy, "record_issued_action", None)
    if not callable(record_fn):
        raise RuntimeError(
            "Action-history policy exposes action_history_horizon but has no callable record_issued_action method."
        )
    record_fn(np.asarray(action, dtype=np.float32))


def get_task_max_steps(task_name: str, max_steps_override: Any = None) -> int:
    if max_steps_override is not None and str(max_steps_override).strip().lower() not in {
        "",
        "none",
        "null",
    }:
        return int(max_steps_override)
    return int(TASK_MAX_STEPS.get(get_base_task_name(task_name), 500))


def rollout_frame_from_views(views: Mapping[str, np.ndarray]) -> np.ndarray:
    left = views["agentview_left"]
    right = views["agentview_right"]
    wrist = views["wrist"]
    wrist_wide = np.repeat(wrist, 2, axis=1)
    bottom = np.concatenate([left, right], axis=1)
    return np.concatenate([wrist_wide, bottom], axis=0)


def save_rollout_video(
    video_path: Path, frames: Sequence[Mapping[str, np.ndarray]], *, fps: int = 20
) -> None:
    if len(frames) == 0:
        return
    import imageio.v2 as imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(rollout_frame_from_views(frame))
    finally:
        writer.close()


def list_to_jsonable(values: Iterable[Any]) -> list[Any]:
    out = []
    for value in values:
        if isinstance(value, np.generic):
            out.append(value.item())
        else:
            out.append(value)
    return out
