"""Run the external RoboTwin evaluator with DreamFree policy callbacks."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set)), resolve=True
    )
    simulator_root = Path(cfg.pop("simulator_root")).expanduser().resolve()
    interpreter = cfg.pop("python", sys.executable)
    output = Path(cfg.pop("output_dir", "./outputs/evaluation/robotwin")).resolve()
    for key in ("task_name", "task_config", "ckpt_setting", "dataset_stats_path"):
        if not cfg.get(key):
            raise ValueError(f"RoboTwin evaluation requires {key}.")
    engine = simulator_root / "script/eval_policy.py"
    if not engine.is_file():
        raise FileNotFoundError(f"RoboTwin evaluator is missing: {engine}")
    for key in (
        "ckpt_setting",
        "dataset_stats_path",
        "text_embedding_cache_dir",
        "model_base_path",
    ):
        if cfg.get(key):
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    cfg["policy_name"] = "dreamfree.evaluation.robotwin"
    output.mkdir(parents=True, exist_ok=True)
    engine_config = output / "config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), engine_config)
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
    subprocess.run(
        [interpreter, str(engine), "--config", str(engine_config)],
        cwd=simulator_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
