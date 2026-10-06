"""Small configuration loader for benchmark evaluators."""

import argparse
import importlib

from omegaconf import OmegaConf


def evaluate_main(environment, argv=None):
    parser = argparse.ArgumentParser(description=f"Evaluate DreamFree on {environment}.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    OmegaConf.resolve(cfg)
    if not cfg.get("ckpt"):
        raise ValueError("ckpt must identify a DreamFree stage 2 checkpoint.")
    module = importlib.import_module(
        f"dreamfree.evaluation.{environment}.eval_{environment}_single"
    )
    run = module.eval_single_process if environment == "libero" else module.main
    run(cfg)
