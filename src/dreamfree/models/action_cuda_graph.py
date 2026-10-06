"""Fixed-shape CUDA graph storage for a prepared action-denoising loop."""

from dataclasses import replace

import torch


def _static_copy(value):
    result = torch.empty_strided(
        value.shape, value.stride(), device=value.device, dtype=value.dtype
    )
    return result.copy_(value)


def _tensor_signature(value):
    return (tuple(value.shape), value.device, value.dtype)


class ActionEulerGraph:
    """Capture once during warmup; refresh every input before each serialized replay."""

    def __init__(
        self, run, action_tokens, conditioning, prefix, mask, *, borrow_conditioning=False
    ):
        if action_tokens.device.type != "cuda" or action_tokens.shape[0] != 1:
            raise ValueError("Action CUDA graph requires CUDA tensors with batch size one.")
        self.noise = _static_copy(action_tokens)
        self.borrow_conditioning = borrow_conditioning
        self.conditioning = (
            conditioning
            if borrow_conditioning
            else replace(
                conditioning,
                block_kv=tuple(
                    tuple(_static_copy(value) for value in pair) for pair in conditioning.block_kv
                ),
            )
        )
        self.prefix = _static_copy(prefix)
        self.mask = None if mask is None else _static_copy(mask)
        self.inputs = self._inputs(self.noise, self.conditioning, self.prefix, self.mask)
        self.signatures = tuple(_tensor_signature(value) for value in self.inputs)
        self.metadata = self._metadata(conditioning)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=action_tokens.device)
        current = torch.cuda.current_stream(action_tokens.device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(self.noise, self.conditioning, self.prefix, self.mask)
            current.wait_stream(stream)
            torch.cuda.synchronize(action_tokens.device)
            with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="global"):
                self.output = run(self.noise, self.conditioning, self.prefix, self.mask)
            current.wait_stream(stream)
        except Exception as exc:
            raise RuntimeError(
                f"Action CUDA graph capture failed (padded_history={mask is not None}): {exc}"
            ) from exc

    @staticmethod
    def _inputs(noise, conditioning, prefix, mask):
        return (
            noise,
            prefix,
            *(value for pair in conditioning.block_kv for value in pair),
            *((mask,) if mask is not None else ()),
        )

    @staticmethod
    def _metadata(conditioning):
        return (
            conditioning.batch_size,
            conditioning.context_length,
            conditioning.hidden_dim,
            conditioning.device,
            conditioning.dtype,
            len(conditioning.block_kv),
        )

    def replay(self, action_tokens, conditioning, prefix, mask):
        values = self._inputs(action_tokens, conditioning, prefix, mask)
        signatures = tuple(_tensor_signature(value) for value in values)
        if (
            self.metadata != self._metadata(conditioning)
            or signatures != self.signatures
            or (mask is None) != (self.mask is None)
        ):
            raise ValueError(
                "Action CUDA graph input shape/device/dtype or conditioning metadata changed; rebuild the policy."
            )
        for index, (destination, source) in enumerate(zip(self.inputs, values)):
            if self.borrow_conditioning and 2 <= index < 2 + 2 * len(self.conditioning.block_kv):
                if destination is not source:
                    raise ValueError(
                        "Borrowed action conditioning storage changed; rebuild the policy."
                    )
                continue
            destination.copy_(source)
        self.graph.replay()
        return self.output.clone()
