from __future__ import annotations

import re

from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)
ROBOTWIN_CAMERA_KEYS = ("images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right")
LIBERO_CAMERA_KEYS = ("observation.images.image", "observation.images.wrist_image")
ROBOCASA_CAMERA_KEYS = (
    "observation.images.agentview_left",
    "observation.images.agentview_right",
    "observation.images.wrist",
)
DROID_CAMERA_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
)
ROBOTWIN_CONCAT_MULTI_CAMERA = "robotwin"
HORIZONTAL_CONCAT_MULTI_CAMERA = "horizontal"
VERTICAL_CONCAT_MULTI_CAMERA = "vertical"
ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA = "robocasa_dreamzero_352"
ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA = "robocasa_dreamzero_448"
ROBOCASA_DREAMZERO_HEIGHT = 352
ROBOCASA_DREAMZERO_WIDTH = 352
ROBOCASA_DREAMZERO_448_HEIGHT = 448
ROBOCASA_DREAMZERO_448_WIDTH = 448
ROBOCASA_DREAMZERO_LAYOUTS = {
    ROBOCASA_DREAMZERO_352_CONCAT_MULTI_CAMERA,
    ROBOCASA_DREAMZERO_448_CONCAT_MULTI_CAMERA,
}
DROID_GRID_CONCAT_MULTI_CAMERA = "droid_grid"
DROID_GRID_HEIGHT = 352
DROID_GRID_WIDTH = 640
DROID_DREAMZERO_ACTION_DIM = 8
DROID_DREAMZERO_PROPRIO_DIM = 8
SUPPORTED_LEROBOT_CODEBASE_VERSIONS = {"v2.0", "v2.1", "v3.0"}
STAGE0_CACHED_LATENTS_DATASET_FORMAT = "stage0_cached_latents"
SUPPORTED_STAGE0_SIGMA_POLICIES = {"uniform_range", "wan_shift_full", "wan_shift_range"}
WAN22_A14B_LATENT_FRAME0_MODES = {"official", "clean_overwrite"}
STAGE0_CHECKPOINT_DIR_RE = re.compile("^step_(\\d+)$")
STAGE0_EVAL_SIGMA_BIN_WIDTH = 0.05
STAGE0_VIDEO_ATTENTION_MASK_MODE = "bidirectional"
