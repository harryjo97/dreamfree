"""MolmoSpaces policy and benchmark configuration; import in its simulator environment."""

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import PromptSampler
from pydantic import Field

from dreamfree.evaluation.droid_sim.molmospaces import GRIPPER_QPOS_OPEN_SCALE, MolmoSpacesClient


class DreamFreeDroidPolicy(InferencePolicy):
    def __init__(self, config, task_type):
        super().__init__(config, task_type)
        self.model = None
        self.prompt_sampler = PromptSampler(
            task_type=task_type,
            prompt_templates=config.policy_config.prompt_templates,
            prompt_object_word_num=config.policy_config.prompt_object_word_num,
        )
        self.reset()

    def prepare_model(self):
        if self.model is None:
            cfg = self.config.policy_config
            self.model = MolmoSpacesClient(
                expected_checkpoint=cfg.checkpoint_path,
                host=cfg.remote_host,
                port=cfg.remote_port,
                replan_steps=cfg.replan_steps,
                timeout_sec=cfg.timeout_sec,
                gripper_qpos_open_scale=cfg.gripper_qpos_open_scale,
            )

    def reset(self):
        self.prompt_sampler.next()
        if self.model is not None:
            self.model.reset()

    def get_action(self, observation):
        if self.task is None:
            raise RuntimeError(
                "MolmoSpaces must attach the current task before requesting actions."
            )
        self.prepare_model()
        instruction = self.task.get_task_description() or self.prompt_sampler.get_prompt(self.task)
        return self.model.get_action(observation, instruction)

    def render(self, observation):
        return None

    def get_info(self):
        return {
            **super().get_info(),
            "policy_name": "dreamfree_droid",
            "policy_checkpoint": self.config.policy_config.checkpoint_path,
            "policy_buffer_length": self.config.policy_config.replan_steps,
            "policy_server_metadata": self.model.server_info if self.model is not None else None,
        }

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None


class DreamFreeDroidPolicyConfig(BasePolicyConfig):
    policy_cls: type = DreamFreeDroidPolicy
    policy_type: str = "learned"
    checkpoint_path: str = ""
    remote_host: str = "127.0.0.1"
    remote_port: int = Field(default=6100, ge=1, le=65535)
    replan_steps: int = Field(default=8, gt=0)
    timeout_sec: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    gripper_qpos_open_scale: float = Field(
        default=GRIPPER_QPOS_OPEN_SCALE, gt=0, allow_inf_nan=False
    )
    prompt_templates: list[str] | None = None
    prompt_object_word_num: int = 1
    eval_save_videos: bool = False
    eval_save_labeled_videos: bool = False
    eval_save_trajectories: bool = False
    eval_save_diagnostics: bool = False
    eval_output_layout: str = "episode_dirs"
    eval_results_html_layout: str = "default"


class DreamFreeDroidEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = Field(default_factory=FrankaRobotConfig)
    policy_config: DreamFreeDroidPolicyConfig = Field(default_factory=DreamFreeDroidPolicyConfig)
    policy_dt_ms: float = 66.0

    def model_post_init(self, context):
        super().model_post_init(context)
        self.robot_config.action_noise_config.enabled = False
