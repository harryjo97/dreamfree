"""Policy-side OSC target client; never owns motors or calls inverse kinematics."""

import time

import numpy as np

if __package__:
    from .common import EefTargetRejected
    from .osc_control import OscFault
    from .osc_servo import endpoint
else:
    from common import EefTargetRejected
    from osc_control import OscFault
    from osc_servo import endpoint


class OscEnv:
    def __init__(self, profile, args, contract):
        import zmq
        from deploy.robot.gym.yam_env import YAMEnv

        # This instance owns only camera subscribers. It never creates followers,
        # publishes joint commands, homes arms, or uses its synthetic debug state.
        self.cameras = YAMEnv(profile, prompt=args.instruction, execute_actions=False)
        self.context = zmq.Context()
        self.address = endpoint(args.ipc_dir)
        self.socket = self._socket()
        self.osc_contract = contract
        self.sequence = 0
        self.closed = False
        self.timeout_ms = max(100, int(1000 * contract["config"]["max_period"] * 3))
        self.last_snapshot = None

    def _socket(self):
        import zmq

        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 100)
        socket.connect(self.address)
        return socket

    def request(self, operation, *, timeout_ms=None, **payload):
        self.socket.send_json(dict(operation=operation, **payload))
        if not self.socket.poll(self.timeout_ms if timeout_ms is None else timeout_ms):
            raise OscFault(
                f"OSC {operation} acknowledgement timed out; target expiry/watchdog must stop the servo."
            )
        response = self.socket.recv_json()
        retryable = response.get("retryable") is True
        if not response.get("ok") and not retryable:
            raise OscFault(f"OSC servo rejected request: {response.get('error')}.")
        if operation != "shutdown":
            age = time.monotonic() - response["feedback_stamp"]
            if not 0 <= age <= self.osc_contract["config"]["feedback_timeout"]:
                raise OscFault("OSC acknowledgement contains stale feedback.")
            self.last_snapshot = response
        if retryable:
            if (
                operation != "target"
                or response.get("ok") is not False
                or response.get("rejected_sequence") != payload.get("sequence")
            ):
                raise OscFault("OSC rejection acknowledgement operation/sequence mismatch.")
            raise EefTargetRejected(response["error"])
        return response

    def reset(self):
        # Capture the present pose; OSC has no automatic home or calibration step.
        self.request("reset", timeout_ms=30000)
        return self.get_obs(), {}

    def get_obs(self, *, state=None):
        observation = self.cameras.get_obs(state=np.zeros(14, dtype=np.float32))
        response = self.request("state")
        observation["state"] = np.asarray(response["state"], dtype=np.float32)
        return observation

    def step_eef(self, action):
        self.sequence += 1
        response = self.request(
            "target",
            action=np.asarray(action).tolist(),
            sequence=self.sequence,
            sent_at=time.monotonic(),
        )
        if response["sequence"] != self.sequence:
            raise OscFault(
                "OSC acknowledgement sequence mismatch; refusing to record action history."
            )
        return (
            np.asarray(response["state"], dtype=np.float32),
            np.asarray(response["accepted_target"], dtype=np.float32),
            response,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            # A previous receive timeout leaves REQ in the wrong state. Use a new
            # connection for explicit shutdown; never republish a torque/target.
            self.socket.close()
            self.socket = self._socket()
            self.request("shutdown", timeout_ms=1000)
        finally:
            self.socket.close()
            self.context.term()
            self.cameras.close()
