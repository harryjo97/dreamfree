from __future__ import annotations

import re

from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)
DREAMFREE_STAGE2_CHECKPOINT_TYPE = "dreamfree_stage2"
DREAMFREE_STAGE2_CHECKPOINT_VERSION = 2
DREAMFREE_STAGE2_SELECTED_DROID_DEFAULT_ACTION_HORIZON = 32
DREAMFREE_STAGE2_CHECKPOINT_DIR_RE = re.compile("^step_(\\d+)$")
READOUT_TYPE_LATENT_QUERY = "latent_query_cross_attention"
SUPPORTED_READOUT_TYPES = {READOUT_TYPE_LATENT_QUERY}
DREAMFREE_STAGE2_REQUIRED_RESUME_FILES = (
    "trainer_state.json",
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)
LIBERO_DREAMFREE_STAGE2_DATASET_NAMES = frozenset(("libero",))
