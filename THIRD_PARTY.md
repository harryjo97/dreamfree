# Third-party components

`src/dreamfree/models/wan22/` contains the Wan model implementation and associated model-loading utilities. Model identifiers point to the upstream Wan checkpoints; their weights are external to this repository.

`src/dreamfree/data/lerobot/lerobot/` and the associated dataset utilities contain adapted Hugging Face LeRobot code. Existing copyright and Apache 2.0 notices are retained in those files.

LIBERO, RoboCasa, RoboTwin, Isaac Sim/Isaac Lab/`sim_evals`, MolmoSpaces, and the physical DROID stack are external runtime integrations. Their code, datasets, models, assets, and licenses are managed separately. No new project-wide license is assigned by this cleanup.

The DROID MolmoSpaces observation mapping and action conversion are adapted from the AWAM/MolmoSpaces evaluation integration. DreamFree supplies its own RPC policy adapter and uses the external MolmoSpaces JSON benchmark evaluator for simulator initialization, scoring, and results.

The ABC/YAM preparation, evaluation, RPC, and robot-control adapters were ported from the local AWAM implementation and adapted to DreamFree's standard Stage 2 and artifact formats. ABC simulation is an external integration with `amazon-far/abc`, pinned to revision `d0832d12651d1b260a652861a14648dc5f3660c7`. ABC-130k data, simulator assets, the YAM model XML, and the physical i2rt stack remain external and retain their respective licenses. `scripts/setup_abc.sh` is an optional installer; none of those assets are bundled here.
