"""Exact image/RPC parity and failure checks for the ABC port."""

import pickle
import threading

import numpy as np
import pytest
import torch

from dreamfree.evaluation.yam import policy as deploy_policy
from dreamfree.evaluation.yam import server as serve_policy
from dreamfree.evaluation.yam_real import client as robot
from dreamfree.evaluation.yam_real import common
from dreamfree.evaluation.yam_real import image_preprocessing as images
from dreamfree.evaluation.yam_real.policy_rpc import PolicyRpcClient


@pytest.mark.parametrize("transport", ["bf16", "fp32"])
def test_packing_matches_training_and_roundtrips_without_raw_images(transport):
    raw = images.startup_observation()
    packed = images.preprocess_observation(raw, transport)
    received = pickle.loads(pickle.dumps(packed, protocol=4))
    common.validate_observation(received)
    actual = deploy_policy.build_image(received, dtype=torch.bfloat16)
    expected = deploy_policy.reference_build_image(raw, dtype=torch.bfloat16)
    assert torch.equal(actual, expected)
    assert "images" not in received and "images" in raw
    assert packed["state"] is raw["state"]
    assert packed[images.PACKED_IMAGE_KEY].nbytes == (995328 if transport == "bf16" else 1990656)
    assert len(pickle.dumps(packed, protocol=4)) < len(pickle.dumps(raw, protocol=4))
    result = serve_policy.validate_image_preprocessing(
        dict(raw_observation=raw, processed_observation=received), torch.bfloat16
    )
    assert result["exact_equal"] and result["transport_exact_equal"]


def test_camera_order_and_normalization_are_preserved():
    raw = images.startup_observation()
    for name, value in zip(common.CAMERAS, (0, 127, 255)):
        raw["images"][name].fill(value)
    image = images.pack_raw_images(raw)
    torch.testing.assert_close(image[:, :, :288], torch.full((1, 3, 288, 384), -1.0))
    torch.testing.assert_close(
        image[:, :, 288:, :192], torch.full((1, 3, 144, 192), 127 / 255 * 2 - 1)
    )
    torch.testing.assert_close(image[:, :, 288:, 192:], torch.ones(1, 3, 144, 192))


@pytest.mark.parametrize(
    "corruption", ["format", "dtype", "shape", "nan", "range", "mixed", "mask"]
)
def test_bad_processed_images_fail_instead_of_falling_back(corruption):
    raw = images.startup_observation()
    packed = images.preprocess_observation(raw)
    if corruption == "format":
        packed[images.PREPROCESSING_KEY] = "unknown"
    elif corruption == "dtype":
        packed[images.PACKED_IMAGE_KEY] = packed[images.PACKED_IMAGE_KEY].astype(np.float32)
    elif corruption == "shape":
        packed[images.PACKED_IMAGE_KEY] = packed[images.PACKED_IMAGE_KEY][:, :-1].copy()
    elif corruption == "nan":
        packed[images.PACKED_IMAGE_KEY][0, 0, 0] = 0x7FC0
    elif corruption == "range":
        packed[images.PACKED_IMAGE_KEY][0, 0, 0] = 0x4000
    elif corruption == "mixed":
        packed["images"] = raw["images"]
    else:
        packed["masks"] = dict.fromkeys(common.CAMERAS, False)
    with pytest.raises(ValueError):
        deploy_policy.build_image(packed, dtype=torch.bfloat16)


def test_repeated_preprocessing_and_bf16_for_fp32_models_fail():
    packed = images.preprocess_observation(images.startup_observation())
    with pytest.raises(ValueError, match="repeated preprocessing"):
        images.preprocess_observation(packed)
    with pytest.raises(ValueError, match="BF16 model input"):
        images.decode_image(packed, dtype=torch.float32)


def test_startup_comparison_detects_changed_pixels():
    raw = images.startup_observation()
    packed = images.preprocess_observation(raw)
    packed[images.PACKED_IMAGE_KEY][0, 200, 200] = 0
    with pytest.raises(ValueError, match="disagrees with the training/server transform"):
        serve_policy.validate_image_preprocessing(
            dict(raw_observation=raw, processed_observation=packed), torch.bfloat16
        )


def test_real_rpc_validates_startup_and_first_live_frame_then_sends_only_packed_images():
    validations = []
    requests = []

    def compare(request):
        validations.append(request)
        return serve_policy.validate_image_preprocessing(request, torch.bfloat16)

    def infer(observation, instruction):
        common.validate_observation(observation)
        requests.append(observation)
        return dict(actions=np.zeros((32, 14), np.float32))

    with serve_policy.SingleClientServer(
        ("127.0.0.1", 0),
        infer_fn=infer,
        info_fn=lambda: dict(image_preprocessing=images.server_contract(torch.bfloat16)),
        validate_image_preprocessing_fn=compare,
    ) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        client = PolicyRpcClient(*server.server_address, timeout_sec=10)
        try:
            preprocessor = images.ClientImagePreprocessor(client, client.info())
            assert len(validations) == 1
            raw = images.startup_observation()
            for _ in range(2):
                client.infer(preprocessor.prepare(raw), raw["prompt"])
            assert len(validations) == 2
            assert all("images" not in request for request in requests)
            assert [v["phase"] for v in preprocessor.metadata["validations"]] == [
                "startup",
                "first_live_frame",
            ]
            assert preprocessor.last_timing["image_validation_seconds"] == 0
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=3)


def test_client_refuses_old_servers_or_failed_comparisons():
    with pytest.raises(RuntimeError, match="update the policy server"):
        images.ClientImagePreprocessor(None, {})
    info = dict(image_preprocessing=images.server_contract(torch.bfloat16))

    class RejectingClient:
        def request(self, request):
            return dict(
                format=request["processed_observation"][images.PREPROCESSING_KEY], exact_equal=False
            )

    with pytest.raises(RuntimeError, match="did not confirm"):
        images.ClientImagePreprocessor(RejectingClient(), info)


def test_client_defaults_to_bf16_and_server_mode_requires_explicit_selection():
    argv = ["--instruction", "task", "--check-only"]
    args = robot.parse_args(argv)
    assert (
        args.yes and args.image_preprocessing == "client" and args.image_transport_dtype == "bf16"
    )
    server_args = robot.parse_args(argv + ["--image-preprocessing", "server"])
    assert server_args.image_transport_dtype is None
    with pytest.raises(SystemExit):
        robot.parse_args(
            argv + ["--image-preprocessing", "server", "--image-transport-dtype", "bf16"]
        )
