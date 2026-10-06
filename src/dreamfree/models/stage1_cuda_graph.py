"""Fixed-shape CUDA graph for current-frame visual feature extraction."""

import torch

from dreamfree.models.action_cuda_graph import _static_copy, _tensor_signature


class Stage1Graph:
    """Refresh observation and instruction tensors before each serialized replay."""

    def __init__(
        self,
        run,
        latents,
        condition,
        context,
        context_mask,
        *,
        borrow_latents=False,
        borrow_output=False,
    ):
        values = (latents, condition, context, context_mask)
        if any(
            value is None or value.device.type != "cuda" or value.shape[0] != 1
            for value in (latents, context, context_mask)
        ) or (
            condition is not None and (condition.device.type != "cuda" or condition.shape[0] != 1)
        ):
            raise ValueError("Stage 1 CUDA graph requires CUDA tensors with batch size one.")
        self.borrow_latents = borrow_latents
        self.borrow_output = borrow_output
        self.borrowed_storage = (
            tuple(
                None if value is None else (value.data_ptr(), tuple(value.stride()))
                for value in values[:2]
            )
            if borrow_latents
            else ()
        )
        self.inputs = tuple(
            None
            if value is None
            else value
            if borrow_latents and index < 2
            else _static_copy(value)
            for (index, value) in enumerate(values)
        )
        self.signatures = tuple(
            None if value is None else _tensor_signature(value) for value in values
        )
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=latents.device)
        current = torch.cuda.current_stream(latents.device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(*self.inputs)
            current.wait_stream(stream)
            torch.cuda.synchronize(latents.device)
            with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="global"):
                self.output = run(*self.inputs)
            current.wait_stream(stream)
        except Exception as exc:
            raise RuntimeError(f"Stage 1 CUDA graph capture failed: {exc}") from exc

    def replay(self, latents, condition, context, context_mask):
        values = (latents, condition, context, context_mask)
        if (
            tuple(None if value is None else _tensor_signature(value) for value in values)
            != self.signatures
        ):
            raise ValueError(
                "Stage 1 CUDA graph input shape/device/dtype or conditioning presence changed; rebuild the policy."
            )
        for index, (destination, source) in enumerate(zip(self.inputs, values)):
            if destination is not None:
                if self.borrow_latents and index < 2:
                    if (source.data_ptr(), tuple(source.stride())) != self.borrowed_storage[index]:
                        raise ValueError(
                            "Borrowed Stage 1 latent storage changed (pointer or strides); rebuild the policy."
                        )
                    continue
                destination.copy_(source)
        self.graph.replay()
        return self.output if self.borrow_output else self.output.clone()
