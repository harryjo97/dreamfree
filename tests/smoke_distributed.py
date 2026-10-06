"""Run with torchrun --standalone --nproc-per-node=2 tests/smoke_distributed.py."""

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dreamfree.training.accumulation import build_fixed_step_gradient_accumulation_plugin


def main():
    accelerator = Accelerator(
        cpu=True, gradient_accumulation_plugin=build_fixed_step_gradient_accumulation_plugin(2)
    )
    if accelerator.num_processes != 2:
        raise RuntimeError("This smoke check requires exactly two processes.")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    # Leave a validation loader at its final batch, then accumulate training normally.
    validation = accelerator.prepare(DataLoader(torch.arange(2), batch_size=1))
    next(iter(validation))
    inputs = torch.tensor([[float(accelerator.process_index + 1)]])
    sync_flags = []
    for _ in range(4):
        with accelerator.accumulate(model):
            accelerator.backward(model(inputs).square().mean())
            sync_flags.append(accelerator.sync_gradients)
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
    assert sync_flags == [False, True, False, True], sync_flags
    actual = accelerator.unwrap_model(model).weight.detach()
    torch.testing.assert_close(actual, torch.full_like(actual, 0.25))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Two-process accumulation matches the global-batch reference.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
