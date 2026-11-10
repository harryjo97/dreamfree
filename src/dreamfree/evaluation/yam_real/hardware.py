"""Shared ABC setup for standalone workstation utilities; no hardware opens on import."""

import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .common import ABC_REVISION, validate_profile
else:
    from common import ABC_REVISION, validate_profile


def add_station_arguments(parser):
    parser.add_argument(
        "--abc-root", type=Path, default=Path(os.environ.get("ABC_ROOT", "./third_party/abc"))
    )
    parser.add_argument("--robot-profile", default=os.environ.get("ROBOT_PROFILE", "bbox_config"))
    parser.add_argument(
        "--ipc-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/data-vol2/jaehyeong/tmp")) / "yam_real_ipc",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the ABC revision and profile without opening hardware.",
    )


def load_profile(args):
    abc_root = args.abc_root.expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(abc_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ABC_REVISION:
        raise ValueError(f"ABC must be at tested revision {ABC_REVISION}, got {revision}.")
    sys.path.insert(0, str(abc_root))
    os.environ["ROBOT_PROFILE"] = args.robot_profile
    if getattr(args, "init_q", None) is not None:
        os.environ["DEPLOY_INIT_Q"] = args.init_q
    from deploy.robot.config import get_i2rt_config

    profile = get_i2rt_config()
    validate_profile(profile)
    return profile


@contextmanager
def station_lock(ipc_dir):
    """Use the rollout's directory lock so these tools cannot overlap its hardware use."""
    import fcntl

    ipc_dir = ipc_dir.expanduser().resolve()
    if len(str(ipc_dir).encode()) + len("/follower_right_state") >= 104:
        raise ValueError("--ipc-dir is too long for a Unix-domain socket.")
    ipc_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(ipc_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another YAM utility or rollout owns this station; stop it first."
            ) from exc
        yield ipc_dir
    finally:
        os.close(fd)


def run_node(*, target, kwargs, ipc_dir):
    import deploy.robot.communication as comms

    comms._IPC_DIR = str(ipc_dir)
    module, _, function = target.partition(":")
    getattr(importlib.import_module(module), function or "run")(**kwargs)


def launch_nodes(specs, ipc_dir):
    from deploy.robot.launch import ProcessSpec, launch

    wrapped = [
        ProcessSpec(
            spec.name,
            f"{__name__}:run_node",
            dict(target=spec.target, kwargs=spec.kwargs, ipc_dir=str(ipc_dir)),
            quiet=spec.quiet,
            terminal_input=spec.terminal_input,
        )
        for spec in specs
    ]
    return launch(wrapped)
