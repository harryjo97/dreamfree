"""Standalone ABC-130K image packing and exact FP32/BF16 RPC transport."""

import time

import numpy as np

if __package__:
    from .common import CAMERAS, validate_observation
else:
    from common import CAMERAS, validate_observation

PREPROCESSING_KEY = "deployment/image_preprocessing"
PACKED_IMAGE_KEY = "deployment/packed_image"
IMAGE_SHAPE = (3, 432, 384)
TRANSPORT_FORMATS = {
    "bf16": "yam_abc130k_crop95_antialias_432x384_normalized_bf16_v1",
    "fp32": "yam_abc130k_crop95_antialias_432x384_normalized_fp32_v1",
}
STORAGE_DTYPES = {"bf16": np.dtype("uint16"), "fp32": np.dtype("float32")}
RAW_IMAGE_BYTES = 3 * 3 * 480 * 640


def require_image_libraries():
    try:
        import torch
        import torchvision
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "YAM client preprocessing requires compatible CPU PyTorch and torchvision. "
            "Use the workstation's matching packages or explicitly select --image-preprocessing server."
        ) from exc
    return torch, torchvision


def pack_raw_images(observation):
    """Match training: float01, center crop, bilinear antialias, pack, then normalize."""
    validate_observation(observation, allow_preprocessed=False)
    torch, torchvision = require_image_libraries()
    transforms = torchvision.transforms.functional
    panels = []
    with torch.inference_mode():
        for name, size in zip(CAMERAS, ((288, 384), (144, 192), (144, 192))):
            view = (
                torch.from_numpy(np.ascontiguousarray(observation["images"][name]))
                .unsqueeze(0)
                .float()
                / 255
            )
            view = transforms.center_crop(view, [456, 608])
            panels.append(
                transforms.resize(
                    view,
                    list(size),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        return (torch.cat((panels[0], torch.cat(panels[1:], dim=-1)), dim=-2) * 2 - 1).contiguous()


def preprocess_observation(observation, transport_dtype="bf16"):
    if transport_dtype not in TRANSPORT_FORMATS:
        raise ValueError(f"Unsupported YAM image transport dtype: {transport_dtype!r}.")
    torch, _ = require_image_libraries()
    image = pack_raw_images(observation)[0]
    # Cast after all FP32 operations, exactly where the original server casts.
    encoded = (
        image.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        if transport_dtype == "bf16"
        else image.numpy()
    )
    result = dict(observation)
    del result["images"]
    result[PREPROCESSING_KEY] = TRANSPORT_FORMATS[transport_dtype]
    result[PACKED_IMAGE_KEY] = np.ascontiguousarray(encoded)
    return result


def validate_packed_image(observation):
    format_name = observation.get(PREPROCESSING_KEY)
    transport_dtype = next(
        (name for name, value in TRANSPORT_FORMATS.items() if value == format_name), None
    )
    if transport_dtype is None:
        raise ValueError(f"Unsupported YAM image preprocessing format: {format_name!r}.")
    if "images" in observation:
        raise ValueError("YAM request must not mix raw images with a packed image.")
    image = observation.get(PACKED_IMAGE_KEY)
    if (
        not isinstance(image, np.ndarray)
        or image.shape != IMAGE_SHAPE
        or image.dtype != STORAGE_DTYPES[transport_dtype]
        or not image.flags.c_contiguous
    ):
        raise ValueError(
            f"Packed YAM image must be contiguous {STORAGE_DTYPES[transport_dtype]} {IMAGE_SHAPE}."
        )
    values = (
        (image.astype(np.uint32) << 16).view(np.float32) if transport_dtype == "bf16" else image
    )
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-6):
        raise ValueError("Packed YAM image must decode to finite normalized pixels in [-1,1].")
    return transport_dtype


def decode_image(observation, *, device="cpu", dtype=None):
    transport_dtype = validate_packed_image(observation)
    torch, _ = require_image_libraries()
    dtype = torch.float32 if dtype is None else dtype
    if transport_dtype == "bf16" and dtype != torch.bfloat16:
        raise ValueError(
            "BF16 YAM transport requires a BF16 model input; select FP32 transport for this model."
        )
    image = observation[PACKED_IMAGE_KEY]
    tensor = (
        torch.from_numpy(image.view(np.int16)).view(torch.bfloat16)
        if transport_dtype == "bf16"
        else torch.from_numpy(image)
    )
    return tensor.unsqueeze(0).to(device=device, dtype=dtype).contiguous()


def server_contract(model_dtype):
    torch, torchvision = require_image_libraries()
    return dict(
        validation_endpoint="validate_image_preprocessing",
        formats=dict(TRANSPORT_FORMATS),
        packed_shape=list(IMAGE_SHAPE),
        model_input_dtype=str(model_dtype).removeprefix("torch."),
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
    )


def startup_observation():
    """A spatially varying probe exercises camera order, crop, resize, and BF16 rounding."""
    rng = np.random.default_rng(713)
    return dict(
        images={name: rng.integers(0, 256, (3, 480, 640), dtype=np.uint8) for name in CAMERAS},
        masks=dict.fromkeys(CAMERAS, True),
        state=np.zeros(14, np.float32),
        prompt="image preprocessing check",
    )


class ClientImagePreprocessor:
    def __init__(self, client, info, *, transport_dtype="bf16", threads=2):
        contract = info.get("image_preprocessing", {})
        if (
            contract.get("validation_endpoint") != "validate_image_preprocessing"
            or contract.get("formats", {}).get(transport_dtype)
            != TRANSPORT_FORMATS.get(transport_dtype)
            or contract.get("packed_shape") != list(IMAGE_SHAPE)
        ):
            raise RuntimeError(
                "Server lacks the requested YAM client preprocessing contract; update the policy server."
            )
        if transport_dtype == "bf16" and contract.get("model_input_dtype") != "bfloat16":
            raise RuntimeError("BF16 image transport requires a bfloat16 policy server.")
        torch, torchvision = require_image_libraries()
        torch.set_num_threads(threads)
        self.client, self.transport_dtype = client, transport_dtype
        self.live_validated = False
        self.last_timing = {}
        self.metadata = dict(
            mode="client",
            format=TRANSPORT_FORMATS[transport_dtype],
            transport_dtype=transport_dtype,
            raw_image_payload_bytes=RAW_IMAGE_BYTES,
            image_payload_bytes=int(np.prod(IMAGE_SHAPE))
            * STORAGE_DTYPES[transport_dtype].itemsize,
            packed_shape=list(IMAGE_SHAPE),
            cpu_threads=threads,
            client_torch_version=str(torch.__version__),
            client_torchvision_version=str(torchvision.__version__),
            server_torch_version=contract.get("torch_version"),
            server_torchvision_version=contract.get("torchvision_version"),
            validations=[],
        )
        raw = startup_observation()
        self.validate(raw, preprocess_observation(raw, transport_dtype), "startup")

    def validate(self, raw, processed, phase):
        started = time.perf_counter()
        response = self.client.request(
            dict(
                endpoint="validate_image_preprocessing",
                raw_observation=dict(
                    images=raw["images"],
                    masks=raw["masks"],
                    state=raw["state"],
                    prompt=raw["prompt"],
                ),
                processed_observation={
                    key: processed[key]
                    for key in (PREPROCESSING_KEY, PACKED_IMAGE_KEY, "masks", "state", "prompt")
                },
            )
        )
        if (
            not isinstance(response, dict)
            or response.get("format") != self.metadata["format"]
            or response.get("exact_equal") is not True
            or response.get("transport_exact_equal") is not True
        ):
            raise RuntimeError(
                f"Server did not confirm exact YAM image preprocessing agreement: {response!r}."
            )
        self.metadata["validations"].append(
            dict(phase=phase, seconds=time.perf_counter() - started, **response)
        )

    def prepare(self, observation):
        started = time.perf_counter()
        result = preprocess_observation(observation, self.transport_dtype)
        self.last_timing = dict(
            client_preprocess_seconds=time.perf_counter() - started, image_validation_seconds=0.0
        )
        if not self.live_validated:
            started = time.perf_counter()
            self.validate(observation, result, "first_live_frame")
            self.last_timing["image_validation_seconds"] = time.perf_counter() - started
            self.live_validated = True
        return result
