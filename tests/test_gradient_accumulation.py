import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from torch.utils.data import DataLoader

from dreamfree.training.accumulation import build_fixed_step_gradient_accumulation_plugin


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.mark.parametrize(
    ("num_steps", "expected_sync_flags"),
    [
        (1, [True, True, True, True]),
        (4, [False, False, False, True, False, False, False, True]),
    ],
)
def test_fixed_step_accumulation_ignores_stranded_validation_state(num_steps, expected_sync_flags):
    accelerator = Accelerator(
        gradient_accumulation_plugin=build_fixed_step_gradient_accumulation_plugin(num_steps),
        step_scheduler_with_optimizer=False,
    )
    val_loader = accelerator.prepare(DataLoader(torch.arange(1), batch_size=1))
    val_iter = iter(val_loader)

    next(val_iter)
    assert accelerator.gradient_state.end_of_dataloader is True

    model = torch.nn.Linear(1, 1)
    sync_flags = []
    for _ in expected_sync_flags:
        with accelerator.accumulate(model):
            sync_flags.append(accelerator.sync_gradients)

    assert sync_flags == expected_sync_flags
    with pytest.raises(StopIteration):
        next(val_iter)


@pytest.mark.parametrize("num_steps", [0, -1])
def test_fixed_step_accumulation_rejects_non_positive_steps(num_steps):
    with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
        build_fixed_step_gradient_accumulation_plugin(num_steps)
