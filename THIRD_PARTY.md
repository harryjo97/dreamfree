# Third-party components

`src/dreamfree/models/wan22/` contains the Wan model implementation and associated model-loading utilities. Model identifiers point to the upstream Wan checkpoints; their weights are external to this repository.

`src/dreamfree/data/lerobot/lerobot/` and the associated dataset utilities contain adapted Hugging Face LeRobot code. Existing copyright and Apache 2.0 notices are retained in those files.

LIBERO, RoboCasa, RoboTwin, IsaacLab/`sim_evals`, and the physical DROID stack are external runtime integrations. Their code, datasets, models, assets, and licenses are managed separately. No new project-wide license is assigned by this cleanup.
