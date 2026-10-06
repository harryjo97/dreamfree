import threading

import numpy as np

from dreamfree.evaluation.droid.policy_rpc import PolicyRpcClient
from dreamfree.evaluation.droid_real.policy_rpc import PolicyRpcServer


def test_simulation_client_and_server_share_rpc_contract():
    seeds = []

    def infer(observation, instruction):
        assert observation["state"].shape == (8,)
        assert instruction == "pick up the cup"
        return {"actions": np.zeros((32, 8), dtype=np.float32)}

    with PolicyRpcServer(
        ("127.0.0.1", 0),
        infer_fn=infer,
        reset_fn=lambda seed: seeds.append(seed),
        info_fn=lambda: {"action_horizon": 32},
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = PolicyRpcClient(*server.server_address)
        try:
            assert client.request({"endpoint": "info"})["action_horizon"] == 32
            reply = client.request(
                {
                    "endpoint": "infer",
                    "observation": {"state": np.zeros(8)},
                    "instruction": "pick up the cup",
                }
            )
            assert reply["actions"].shape == (32, 8)
            assert client.request({"endpoint": "reset", "episode_seed": 7})["status"] == "reset"
            assert seeds == [7]
        finally:
            client.close()
            server.shutdown()
            thread.join()
