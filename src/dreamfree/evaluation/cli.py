"""Small configuration loader for benchmark evaluators."""

import argparse
import importlib

from omegaconf import OmegaConf

from dreamfree.config_names import INFERENCE_RENAMES, rename_fields


def evaluate_main(environment, argv=None):
    parser = argparse.ArgumentParser(description=f"Evaluate DreamFree on {environment}.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    documents = []
    for document in (OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set)):
        values = rename_fields(OmegaConf.to_container(document), INFERENCE_RENAMES)
        if "eval_num_inference_steps" in values:
            raise ValueError(
                "Use EVALUATION.num_inference_steps instead of eval_num_inference_steps."
            )
        if "EVALUATION" in values:
            values["EVALUATION"] = rename_fields(values["EVALUATION"], INFERENCE_RENAMES)
        documents.append(OmegaConf.create(values))
    cfg = OmegaConf.merge(*documents)
    OmegaConf.resolve(cfg)
    if not cfg.get("checkpoint_path"):
        raise ValueError("checkpoint_path must identify a DreamFree stage 2 checkpoint.")
    module = importlib.import_module(
        f"dreamfree.evaluation.{environment}.eval_{environment}_single"
    )
    run = module.eval_single_process if environment == "libero" else module.main
    run(cfg)
