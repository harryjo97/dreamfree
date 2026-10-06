from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor


class FlowUniPCMultistepScheduler:
    """Dependency-free Wan Flow-UniPC scheduler.

    This is a local port of Wan2.2's `FlowUniPCMultistepScheduler` with the
    diffusers mixins removed so the experimental environment does not need a new
    dependency.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float | None = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] | None = None,
        solver_p: Any | None = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: str | None = "zero",
    ) -> None:
        if int(num_train_timesteps) <= 0:
            raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}.")
        if int(solver_order) <= 0:
            raise ValueError(f"solver_order must be positive, got {solver_order}.")
        if str(prediction_type) != "flow_prediction":
            raise ValueError(
                f"FlowUniPCMultistepScheduler requires prediction_type='flow_prediction', got {prediction_type!r}."
            )
        if str(solver_type) not in {"bh1", "bh2"}:
            raise NotImplementedError(
                f"{solver_type!r} is not implemented for FlowUniPCMultistepScheduler."
            )
        if final_sigmas_type not in {"zero", "sigma_min"}:
            raise ValueError(
                f"final_sigmas_type must be 'zero' or 'sigma_min', got {final_sigmas_type!r}."
            )
        self.config = SimpleNamespace(
            num_train_timesteps=int(num_train_timesteps),
            solver_order=int(solver_order),
            prediction_type=str(prediction_type),
            shift=1.0 if shift is None else float(shift),
            use_dynamic_shifting=bool(use_dynamic_shifting),
            thresholding=bool(thresholding),
            dynamic_thresholding_ratio=float(dynamic_thresholding_ratio),
            sample_max_value=float(sample_max_value),
            solver_type=str(solver_type),
            lower_order_final=bool(lower_order_final),
            timestep_spacing=str(timestep_spacing),
            steps_offset=int(steps_offset),
            final_sigmas_type=str(final_sigmas_type),
        )
        self.predict_x0 = bool(predict_x0)
        self.disable_corrector = (
            [] if disable_corrector is None else [int(item) for item in disable_corrector]
        )
        self.solver_p = solver_p
        self.num_inference_steps: int | None = None
        alphas = np.linspace(
            1, 1 / self.config.num_train_timesteps, self.config.num_train_timesteps
        )[::-1].copy()
        sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)
        if not self.config.use_dynamic_shifting:
            shift_value = self.config.shift
            sigmas = shift_value * sigmas / (1 + (shift_value - 1) * sigmas)
        self.sigmas = sigmas.to("cpu")
        self.timesteps = self.sigmas * self.config.num_train_timesteps
        self.sigma_min = float(self.sigmas[-1].item())
        self.sigma_max = float(self.sigmas[0].item())
        self.model_outputs: list[torch.Tensor | None] = [None] * self.config.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | np.ndarray | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ) -> None:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("mu is required when use_dynamic_shifting=True.")
        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required when sigmas is None.")
            if int(num_inference_steps) <= 0:
                raise ValueError(
                    f"num_inference_steps must be positive, got {num_inference_steps}."
                )
            sigmas_arr = np.linspace(
                self.sigma_max, self.sigma_min, int(num_inference_steps) + 1
            ).copy()[:-1]
        else:
            sigmas_arr = np.asarray(sigmas, dtype=np.float32).copy()
            if sigmas_arr.ndim != 1 or sigmas_arr.size == 0:
                raise ValueError(
                    f"sigmas must be a non-empty 1D sequence, got shape {sigmas_arr.shape}."
                )
        if self.config.use_dynamic_shifting:
            sigmas_arr = self.time_shift(float(mu), 1.0, sigmas_arr)
        else:
            shift_value = self.config.shift if shift is None else float(shift)
            sigmas_arr = shift_value * sigmas_arr / (1 + (shift_value - 1) * sigmas_arr)
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(f"Unsupported final_sigmas_type={self.config.final_sigmas_type!r}.")
        timesteps = sigmas_arr * float(self.config.num_train_timesteps)
        self.sigmas = torch.from_numpy(
            np.concatenate([sigmas_arr, [sigma_last]]).astype(np.float32)
        ).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = int(len(timesteps))
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p is not None:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)
        self._step_index = None
        self._begin_index = None

    @staticmethod
    def time_shift(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        (batch_size, channels, *remaining_dims) = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))
        abs_sample = sample.abs()
        threshold = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        threshold = torch.clamp(threshold, min=1, max=self.config.sample_max_value).unsqueeze(1)
        sample = torch.clamp(sample, -threshold, threshold) / threshold
        return sample.reshape(batch_size, channels, *remaining_dims).to(dtype)

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * float(self.config.num_train_timesteps)

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (1 - sigma, sigma)

    @staticmethod
    def _stack_scalar_terms(
        values: list[torch.Tensor | float], *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        terms = [
            value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=device, dtype=dtype)
            for value in values
        ]
        return torch.stack(terms, dim=0)

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args: Any,
        sample: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument.")
        if self.step_index is None:
            raise RuntimeError("step_index is not initialized.")
        sigma_t = self.sigmas[self.step_index].to(device=sample.device, dtype=sample.dtype)
        if self.predict_x0:
            x0_pred = sample - sigma_t * model_output
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred
        epsilon = sample - (1 - sigma_t) * model_output
        if self.config.thresholding:
            x0_pred = self._threshold_sample(sample - sigma_t * model_output)
            epsilon = model_output + x0_pred
        return epsilon

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args: Any,
        sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument.")
        if order is None:
            if len(args) > 2:
                order = int(args[2])
            else:
                raise ValueError("missing `order` as a required keyword argument.")
        if self.step_index is None:
            raise RuntimeError("step_index is not initialized.")
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("The current model output is missing.")
        if self.solver_p is not None:
            return self.solver_p.step(model_output, self.timestep_list[-1], sample).prev_sample
        output_dtype = sample.dtype
        compute_dtype = (
            torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        )
        sample_compute = sample.to(dtype=compute_dtype)
        m0 = m0.to(dtype=compute_dtype)
        sigma_t = self.sigmas[self.step_index + 1].to(device=sample.device, dtype=compute_dtype)
        sigma_s0 = self.sigmas[self.step_index].to(device=sample.device, dtype=compute_dtype)
        (alpha_t, sigma_t) = self._sigma_to_alpha_sigma_t(sigma_t)
        (alpha_s0, sigma_s0) = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        rks: list[torch.Tensor | float] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, int(order)):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            if mi is None:
                raise RuntimeError(f"Missing previous model output for order={order}.")
            mi = mi.to(dtype=compute_dtype)
            (alpha_si, sigma_si) = self._sigma_to_alpha_sigma_t(
                self.sigmas[si].to(device=sample.device, dtype=compute_dtype)
            )
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(1.0)
        solve_dtype = torch.float32
        rks_tensor = self._stack_scalar_terms(rks, device=sample.device, dtype=solve_dtype)
        r_matrix: list[torch.Tensor] = []
        b_values: list[torch.Tensor] = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
        for i in range(1, int(order) + 1):
            r_matrix.append(torch.pow(rks_tensor, i - 1))
            b_values.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        r_matrix_tensor = torch.stack(r_matrix).to(dtype=solve_dtype)
        b_tensor = self._stack_scalar_terms(b_values, device=sample.device, dtype=solve_dtype)
        if d1s:
            d1s_tensor = torch.stack(d1s, dim=1)
            if int(order) == 2:
                rhos_p = torch.tensor([0.5], dtype=compute_dtype, device=sample.device)
            else:
                rhos_p = (
                    torch.linalg.solve(r_matrix_tensor[:-1, :-1], b_tensor[:-1])
                    .to(sample.device)
                    .to(compute_dtype)
                )
        else:
            d1s_tensor = None
        if self.predict_x0:
            x_t = sigma_t / sigma_s0 * sample_compute - alpha_t * h_phi_1 * m0
            if d1s_tensor is not None:
                x_t = x_t - alpha_t * b_h * torch.einsum("k,bkc...->bc...", rhos_p, d1s_tensor)
        else:
            x_t = alpha_t / alpha_s0 * sample_compute - sigma_t * h_phi_1 * m0
            if d1s_tensor is not None:
                x_t = x_t - sigma_t * b_h * torch.einsum("k,bkc...->bc...", rhos_p, d1s_tensor)
        return x_t.to(output_dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args: Any,
        last_sample: torch.Tensor | None = None,
        this_sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as a required keyword argument.")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as a required keyword argument.")
        if order is None:
            if len(args) > 3:
                order = int(args[3])
            else:
                raise ValueError("missing `order` as a required keyword argument.")
        if self.step_index is None:
            raise RuntimeError("step_index is not initialized.")
        m0 = self.model_outputs[-1]
        if m0 is None:
            raise RuntimeError("The current model output is missing.")
        output_dtype = last_sample.dtype
        compute_dtype = (
            torch.float32
            if last_sample.dtype in (torch.float16, torch.bfloat16)
            else last_sample.dtype
        )
        m0 = m0.to(dtype=compute_dtype)
        model_t = this_model_output.to(dtype=compute_dtype)
        last_sample_compute = last_sample.to(dtype=compute_dtype)
        sigma_t = self.sigmas[self.step_index].to(device=this_sample.device, dtype=compute_dtype)
        sigma_s0 = self.sigmas[self.step_index - 1].to(
            device=this_sample.device, dtype=compute_dtype
        )
        (alpha_t, sigma_t) = self._sigma_to_alpha_sigma_t(sigma_t)
        (alpha_s0, sigma_s0) = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        rks: list[torch.Tensor | float] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, int(order)):
            si = self.step_index - (i + 1)
            mi = self.model_outputs[-(i + 1)]
            if mi is None:
                raise RuntimeError(f"Missing previous model output for corrector order={order}.")
            mi = mi.to(dtype=compute_dtype)
            (alpha_si, sigma_si) = self._sigma_to_alpha_sigma_t(
                self.sigmas[si].to(device=this_sample.device, dtype=compute_dtype)
            )
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(1.0)
        solve_dtype = torch.float32
        rks_tensor = self._stack_scalar_terms(rks, device=this_sample.device, dtype=solve_dtype)
        r_matrix: list[torch.Tensor] = []
        b_values: list[torch.Tensor] = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
        for i in range(1, int(order) + 1):
            r_matrix.append(torch.pow(rks_tensor, i - 1))
            b_values.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        r_matrix_tensor = torch.stack(r_matrix).to(dtype=solve_dtype)
        b_tensor = self._stack_scalar_terms(b_values, device=this_sample.device, dtype=solve_dtype)
        d1s_tensor = torch.stack(d1s, dim=1) if d1s else None
        if int(order) == 1:
            rhos_c = torch.tensor([0.5], dtype=compute_dtype, device=this_sample.device)
        else:
            rhos_c = (
                torch.linalg.solve(r_matrix_tensor, b_tensor)
                .to(this_sample.device)
                .to(compute_dtype)
            )
        if self.predict_x0:
            x_t = sigma_t / sigma_s0 * last_sample_compute - alpha_t * h_phi_1 * m0
            corr_res = (
                0
                if d1s_tensor is None
                else torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s_tensor)
            )
            x_t = x_t - alpha_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        else:
            x_t = alpha_t / alpha_s0 * last_sample_compute - sigma_t * h_phi_1 * m0
            corr_res = (
                0
                if d1s_tensor is None
                else torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s_tensor)
            )
            x_t = x_t - sigma_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        return x_t.to(output_dtype)

    def index_for_timestep(
        self, timestep: int | torch.Tensor, schedule_timesteps: torch.Tensor | None = None
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(schedule_timesteps.device)
        indices = (schedule_timesteps == timestep).nonzero()
        if int(indices.numel()) == 0:
            raise ValueError(f"timestep={timestep} is not in the current schedule.")
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self.begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        del generator
        if self.num_inference_steps is None:
            raise ValueError("Run set_timesteps before calling step.")
        if self.step_index is None:
            self._init_step_index(timestep)
        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and (self.last_sample is not None)
        )
        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = (
            timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep)
        )
        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        if self.this_order <= 0:
            raise RuntimeError(f"Invalid UniPC order={self.this_order}.")
        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output, sample=sample, order=self.this_order
        )
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    @staticmethod
    def scale_model_input(sample: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return sample

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * int(timesteps.shape[0])
        else:
            step_indices = [self.begin_index] * int(timesteps.shape[0])
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        (alpha_t, sigma_t) = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self) -> int:
        return self.config.num_train_timesteps
