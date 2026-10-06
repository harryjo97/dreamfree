"""Exact DROID crop/resize shared by the server and Python 3.7 robot client.

Copy this file beside run_dreamfree_stage2_robot.py on the robot laptop. PyTorch is
imported only when preprocessing is used; no DreamFree installation is needed.
"""

import time

import numpy as np

IMAGE_KEYS = (
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
    "observation/wrist_image_left",
)
PREPROCESSING_KEY = "deployment/image_preprocessing"
PREPROCESSING_FORMAT = "droid_crop95_bilinear_176x320_float32_chw_v1"
TRANSPORT_FORMATS = {
    "fp32": PREPROCESSING_FORMAT,
    "bf16": "droid_crop95_bilinear_176x320_bfloat16_normalized_chw_v1",
    "uint8": "droid_crop95_bilinear_176x320_uint8_rgb_chw_v1",
}
STORAGE_DTYPES = {
    "fp32": np.dtype("float32"),
    "bf16": np.dtype("uint16"),
    "uint8": np.dtype("uint8"),
}
IMAGE_SIZE = (176, 320)


def transport_dtype_from_format(format_name):
    for dtype, name in TRANSPORT_FORMATS.items():
        if name == format_name:
            return dtype
    raise ValueError("Unsupported DROID image preprocessing format: {!r}.".format(format_name))


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Client image preprocessing requires CPU PyTorch in the DROID environment. Install a compatible build, or explicitly select --image-preprocessing server."
        ) from exc
    return torch


def rgb_to_chw_frame(rgb):
    """Apply the original server conversion before cropping or resizing.

    Args:
        rgb: RGB array/tensor, HWC or CHW; uint8 pixels are divided by 255.

    Returns:
        Float32 tensor shaped (1, 3, H, W), preserving the original arithmetic.
    """
    torch = require_torch()
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach()
    else:
        tensor = torch.as_tensor(np.asarray(rgb))
    if tensor.ndim != 3:
        raise ValueError(
            "DROID RGB frame must be 3D HWC or CHW, got {}.".format(tuple(tensor.shape))
        )
    if int(tensor.shape[-1]) == 3:
        tensor = tensor.permute(2, 0, 1)
    elif int(tensor.shape[0]) != 3:
        raise ValueError(
            "DROID RGB frame must have 3 channels, got {}.".format(tuple(tensor.shape))
        )
    tensor = tensor.contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    elif torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
    else:
        raise TypeError("DROID RGB frame must be uint8 or floating, got {}.".format(tensor.dtype))
    return tensor.unsqueeze(0)


def center_crop_resize_frame(frame, *, size=IMAGE_SIZE, crop_scale=0.95):
    """Use the same crop rounding and interpolation on either machine.

    Args:
        frame: Float32 camera tensor shaped (1, 3, H, W).
        size: Output (height, width), normally (176, 320).
        crop_scale: Fraction retained along each axis by the center crop.

    Returns:
        Float32 resized tensor; no rounding back to uint8 is performed.
    """
    torch = require_torch()
    if frame.ndim != 4 or int(frame.shape[1]) != 3:
        raise ValueError(
            "Expected DROID camera frame [1, 3, H, W], got {}.".format(tuple(frame.shape))
        )
    (_, _, height, width) = frame.shape
    crop_h = max(1, min(int(height), int(round(height * float(crop_scale)))))
    crop_w = max(1, min(int(width), int(round(width * float(crop_scale)))))
    top = max((int(height) - crop_h) // 2, 0)
    left = max((int(width) - crop_w) // 2, 0)
    cropped = frame[..., top : top + crop_h, left : left + crop_w]
    return torch.nn.functional.interpolate(
        cropped, size=size, mode="bilinear", align_corners=False, antialias=True
    )


def encode_resized_frame(frame, transport_dtype):
    """Encode FP32 CHW RGB; BF16 follows normalization, uint8 precedes it."""
    torch = require_torch()
    if transport_dtype not in TRANSPORT_FORMATS:
        raise ValueError("Unsupported image transport dtype: {!r}.".format(transport_dtype))
    if frame.dtype != torch.float32 or not torch.isfinite(frame).all():
        raise ValueError("Resized images must contain finite float32 RGB values.")
    if transport_dtype == "fp32":
        return frame.contiguous().numpy()
    if transport_dtype == "uint8":
        return (frame * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous().numpy()
    normalized = frame.clamp(0.0, 1.0) * 2.0 - 1.0
    return normalized.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def preprocess_observation(observation, transport_dtype="fp32"):
    """Crop/resize in FP32, then encode the explicitly selected transport format.

    Args:
        observation: Raw camera images, robot state, and optional action history.
        transport_dtype: fp32 RGB, normalized bf16 bits, or uint8 RGB.
            The low-level default retains compatibility with existing FP32 callers;
            robot clients explicitly select bf16 by default.

    Returns:
        Shallow request copy with three (3, 176, 320) encoded images and a format
        marker. State/history are unchanged; original images remain available.
    """
    if PREPROCESSING_KEY in observation:
        raise ValueError("Image preprocessing requires raw images; refusing to preprocess twice.")
    result = dict(observation)
    for key in IMAGE_KEYS:
        frame = rgb_to_chw_frame(observation[key])
        if frame.device.type != "cpu":
            raise ValueError("Client image preprocessing must run on CPU.")
        result[key] = encode_resized_frame(center_crop_resize_frame(frame)[0], transport_dtype)
    result[PREPROCESSING_KEY] = TRANSPORT_FORMATS[transport_dtype]
    return result


def processed_frames(observation):
    """Restore transmitted values without repeating the completed operations.

    Args:
        observation: Marked request containing contiguous encoded CHW images.

    Returns:
        Three float32 (1, 3, 176, 320) tensors for server grid assembly.
        BF16 is already normalized; FP32 and uint8 decode to RGB in [0,1].
    """
    torch = require_torch()
    transport_dtype = transport_dtype_from_format(observation.get(PREPROCESSING_KEY))
    frames = []
    for key in IMAGE_KEYS:
        value = observation[key]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != STORAGE_DTYPES[transport_dtype]
            or value.shape != (3,) + IMAGE_SIZE
            or (not value.flags.c_contiguous)
            or (not np.isfinite(value).all())
        ):
            raise ValueError(
                "{} must be finite contiguous {} CHW with shape (3, 176, 320).".format(
                    key, STORAGE_DTYPES[transport_dtype]
                )
            )
        if transport_dtype == "bf16":
            frame = torch.from_numpy(value.view(np.int16)).view(torch.bfloat16).float()
        elif transport_dtype == "uint8":
            frame = torch.from_numpy(value).float() / 255.0
        else:
            frame = torch.from_numpy(value)
        if not torch.isfinite(frame).all():
            raise ValueError("{} decodes to non-finite {} pixels.".format(key, transport_dtype))
        if transport_dtype == "bf16" and (torch.any(frame < -1) or torch.any(frame > 1)):
            raise ValueError("{} normalized pixels must be in [-1,1].".format(key))
        frames.append(frame.unsqueeze(0))
    return frames


class ClientImagePreprocessor(object):
    """Validate each camera-shape configuration before its first policy request."""

    def __init__(self, client, server_info, transport_dtype="bf16"):
        if transport_dtype not in TRANSPORT_FORMATS:
            raise ValueError("Unsupported image transport dtype: {!r}.".format(transport_dtype))
        contract = server_info.get("image_preprocessing", {})
        if (
            not isinstance(contract, dict)
            or contract.get("validation_endpoint") != "validate_image_preprocessing"
        ):
            raise RuntimeError(
                "Server does not support exact client image preprocessing; update the server."
            )
        format_name = TRANSPORT_FORMATS[transport_dtype]
        formats = contract.get("formats", {"fp32": contract.get("format")})
        if not isinstance(formats, dict) or formats.get(transport_dtype) != format_name:
            raise RuntimeError(
                "Server does not support {} image transport; update the server or explicitly select --image-transport-dtype fp32.".format(
                    transport_dtype
                )
            )
        if transport_dtype == "bf16" and contract.get("model_input_dtype") != "bfloat16":
            raise RuntimeError(
                "BF16 image transport requires a bfloat16 model input; select --image-transport-dtype fp32 for this server."
            )
        torch = require_torch()
        try:
            resized = center_crop_resize_frame(torch.zeros((1, 3, 8, 8), dtype=torch.float32))
            encode_resized_frame(resized[0], transport_dtype)
        except (TypeError, RuntimeError, NotImplementedError) as exc:
            raise RuntimeError(
                "PyTorch {} cannot perform the required resize/{} encoding: {}".format(
                    torch.__version__, transport_dtype, exc
                )
            ) from exc
        self.client = client
        self.transport_dtype = transport_dtype
        self.validated_shapes = set()
        self.metadata = {
            "mode": "client",
            "format": format_name,
            "transport_dtype": transport_dtype,
            "normalization": "normalized_minus1_1" if transport_dtype == "bf16" else "rgb_0_1",
            "lossy": transport_dtype == "uint8",
            "model_input_dtype": contract.get("model_input_dtype"),
            "client_torch_version": str(torch.__version__),
            "server_torch_version": contract.get("torch_version"),
            "transmitted_shape_per_camera": [3, 176, 320],
            "transmitted_dtype": str(STORAGE_DTYPES[transport_dtype]),
            "image_payload_bytes": 3 * 3 * 176 * 320 * STORAGE_DTYPES[transport_dtype].itemsize,
            "preprocess_seconds_total": 0.0,
            "validation_seconds_total": 0.0,
            "validations": [],
        }

    def prepare(self, observation, client_timing=None):
        start = time.perf_counter()
        result = preprocess_observation(observation, self.transport_dtype)
        elapsed = time.perf_counter() - start
        self.metadata["preprocess_seconds_total"] += elapsed
        if client_timing is not None:
            field = "client_preprocess_seconds"
            client_timing[field] = client_timing.get(field, 0.0) + elapsed
        shapes = (self.metadata["format"],) + tuple(
            tuple(np.asarray(observation[key]).shape) for key in IMAGE_KEYS
        )
        if shapes not in self.validated_shapes:
            record = {
                "raw_shapes": [list(shape) for shape in shapes[1:]],
                "status": "failed",
                "raw_image_payload_bytes": sum(
                    np.asarray(observation[key]).nbytes for key in IMAGE_KEYS
                ),
            }
            self.metadata["validations"].append(record)
            start = time.perf_counter()
            try:
                response = self.client.request(
                    {
                        "endpoint": "validate_image_preprocessing",
                        "raw_observation": {key: observation[key] for key in IMAGE_KEYS},
                        "processed_observation": {
                            key: result[key] for key in IMAGE_KEYS + (PREPROCESSING_KEY,)
                        },
                        "client_torch_version": self.metadata["client_torch_version"],
                    }
                )
                valid = (
                    isinstance(response, dict) and response.get("format") == self.metadata["format"]
                )
                if self.transport_dtype == "fp32":
                    valid = valid and response.get("exact_equal") is True
                else:
                    valid = valid and response.get("transport_exact_equal") is True
                    if self.transport_dtype == "bf16":
                        valid = (
                            valid
                            and response.get("exact_equal") is True
                            and (response.get("model_input_dtype") == "bfloat16")
                        )
                    else:
                        valid = valid and response.get("lossy") is True
                if not valid:
                    raise RuntimeError(
                        "Server did not confirm exact image preprocessing parity: {!r}".format(
                            response
                        )
                    )
                record.update(response)
                record["status"] = "passed"
                self.validated_shapes.add(shapes)
            except Exception as exc:
                record["error"] = str(exc)
                raise
            finally:
                record["seconds"] = time.perf_counter() - start
                self.metadata["validation_seconds_total"] += record["seconds"]
        return result
