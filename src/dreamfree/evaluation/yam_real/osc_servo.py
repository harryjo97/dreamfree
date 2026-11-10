"""Continuous workstation OSC service and hardware-free servo state machine."""

import time
from collections import deque
from pathlib import Path

import numpy as np

if __package__:
    from .common import EefTargetRejected, write_json
    from .osc_control import HISTORY_CONTRACT, ArmOsc, OscFault, encode_pose, limit_target
    from .osc_driver import SynchronousYamDriver, open_pinned_can, prepare_setup
else:
    from common import EefTargetRejected, write_json
    from osc_control import HISTORY_CONTRACT, ArmOsc, OscFault, encode_pose, limit_target
    from osc_driver import SynchronousYamDriver, open_pinned_can, prepare_setup


def endpoint(ipc_dir):
    return "ipc://" + str(Path(ipc_dir) / "awam_osc")


class OscServo:
    """Single owner of both arm transports; GPU requests only replace EEF targets."""

    def __init__(self, controllers, drivers, config, *, clock=time.monotonic, rejection_retries=0):
        if len(controllers) != 2 or len(drivers) != 2:
            raise ValueError("OSC servo requires left and right arms in that order.")
        self.controllers, self.drivers, self.config, self.clock = (
            controllers,
            drivers,
            config,
            clock,
        )
        if type(rejection_retries) is not int or rejection_retries < 0:
            raise ValueError("OSC rejection_retries must be a nonnegative integer.")
        self.rejection_retries, self.consecutive_rejections = rejection_retries, 0
        self.fault = None
        self.active = False
        self.sequence = -1
        self.last_request_sequence = -1
        self.periods, self.compute_times = deque(maxlen=10000), deque(maxlen=10000)
        self.count, self.misses, self.consecutive_misses = 0, 0, 0
        self.started = self.last_tick = self.target_stamp = self.last_target_update = None
        self.stopped = None
        self.target = None
        self.target_report = None
        self.feedback = None

    def start(self):
        try:
            self.feedback = [driver.enable() for driver in self.drivers]
            self.active = True
            self.started = self.last_tick = self.clock()
            self.reset_target(self.started)
        except BaseException as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            self.stop()
            raise

    def measured_action(self):
        poses = []
        for arm, (q, _, _) in zip(self.controllers, self.feedback):
            pose = arm.dynamics.fk.forward(q[:6])
            poses.append(encode_pose(pose[:3, 3], pose[:3, :3], q[6]))
        return np.concatenate(poses)

    def reset_target(self, now):
        self.check_health(now, check_target=False)
        self.target = self.measured_action()
        # Endpoint quantization may put measured aperture just outside [0,1].
        # Preserve that observation while bounding the captured hold target.
        grip = self.target[[9, 19]].copy()
        self.target[[9, 19]] = np.clip(grip, 0, 1)
        self.target_stamp = self.last_target_update = now
        self.target_report = [
            dict(
                position_scale=1.0,
                rotation_scale=1.0,
                gripper_limited=bool(value < 0 or value > 1),
                rotation_orthonormalized=False,
            )
            for value in grip
        ]
        return self.target.copy()

    def check_health(self, now, *, check_target=True):
        if self.fault is not None or not self.active:
            raise OscFault(f"OSC servo is inactive or faulted: {self.fault}.")
        if not np.isfinite(now):
            raise OscFault("Nonfinite OSC clock.")
        for side, (_, _, stamps) in zip(("left", "right"), self.feedback):
            age = now - np.asarray(stamps)
            if (
                age.shape != (7,)
                or not np.isfinite(age).all()
                or np.any(age < 0)
                or np.any(age > self.config.feedback_timeout)
            ):
                raise OscFault(f"OSC {side} feedback stale or timestamps invalid.")
        if check_target and (
            now - self.target_stamp > self.config.target_timeout or now < self.target_stamp
        ):
            raise OscFault(
                "OSC Cartesian target expired; inference/target publisher stopped updating."
            )

    def submit(self, action, sequence, sent_at, now):
        try:
            self.check_health(now)
            if type(sequence) is not int or sequence <= self.last_request_sequence:
                raise OscFault(
                    "OSC target sequence must increase; duplicate/replayed targets are rejected."
                )
            if not np.isfinite(sent_at) or not 0 <= now - sent_at <= self.config.target_timeout:
                raise OscFault("OSC target message expired or clock is invalid.")
            self.last_request_sequence = sequence
            # A fault while tracking the accepted target cannot be repaired by
            # replacing it. Only rejection of a new candidate is recoverable.
            for i, (arm, (q, dq, _)) in enumerate(zip(self.controllers, self.feedback)):
                try:
                    arm.validate_target(q, dq, self.target[i * 10 : (i + 1) * 10])
                except EefTargetRejected as exc:
                    raise OscFault(f"OSC active target tracking bound violated: {exc}") from exc
            target, report = limit_target(
                action, self.target, now - self.last_target_update, self.config
            )
            for i, (arm, (q, dq, _)) in enumerate(zip(self.controllers, self.feedback)):
                arm.validate_target(q, dq, target[i * 10 : (i + 1) * 10])
            self.target, self.target_report = target, report
            self.sequence, self.target_stamp, self.last_target_update = sequence, sent_at, now
            self.consecutive_rejections = 0
        except BaseException as exc:
            if isinstance(exc, EefTargetRejected):
                self.consecutive_rejections += 1
                if self.consecutive_rejections <= self.rejection_retries:
                    # Keep both the accepted target and its original expiry.
                    raise
            self.fault = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, EefTargetRejected):
                self.fault = (
                    f"OSC target rejection retry limit exhausted "
                    f"({self.rejection_retries} retries): {exc}"
                )
            self.stop()
            if isinstance(exc, EefTargetRejected):
                raise OscFault(self.fault) from exc
            raise

    def tick(self, now=None):
        now = self.clock() if now is None else now
        try:
            self.check_health(now)
            dt = now - self.last_tick if self.count else 1 / self.config.rate_hz
            if not 0 < dt <= self.config.max_period:
                raise OscFault(f"OSC missed absolute servo deadline: period={dt:.6f}s.")
            missed = dt > 1.25 / self.config.rate_hz
            self.misses += int(missed)
            self.consecutive_misses = self.consecutive_misses + 1 if missed else 0
            if self.consecutive_misses >= self.config.max_consecutive_misses:
                raise OscFault("OSC consecutive servo deadline misses exceeded configured bound.")
            begin = self.clock()
            commands, reports = [], []
            # Compute both arms before I/O. CAN transmission itself is sequential,
            # not atomic: any partial exchange failure disables both arms.
            for index, (controller, (q, dq, _)) in enumerate(zip(self.controllers, self.feedback)):
                torque, report = controller.compute(
                    q, dq, self.target[index * 10 : (index + 1) * 10], dt
                )
                commands.append(torque)
                reports.append(report)
            self.check_health(self.clock())
            if self.clock() - now > self.config.max_period:
                raise OscFault("OSC computation exceeded deadline before motor send.")
            self.feedback = [
                driver.exchange(torque) for driver, torque in zip(self.drivers, commands)
            ]
            for driver, report in zip(self.drivers, reports):
                if hasattr(driver, "last_wire_torque"):
                    report["wire_torque"] = driver.last_wire_torque.tolist()
            finished = self.clock()
            if finished - now > self.config.max_period:
                raise OscFault("OSC motor transaction exceeded servo deadline.")
            self.check_health(finished)
            self.compute_times.append(finished - begin)
            self.periods.append(dt)
            self.last_tick = now
            self.count += 1
            return dict(
                state=np.concatenate([s[0] for s in self.feedback]).tolist(),
                measured_pose=self.measured_action().tolist(),
                accepted_target=self.target.tolist(),
                target_limiting=self.target_report,
                sequence=self.sequence,
                reports=reports,
                feedback_stamp=float(min(np.min(s[2]) for s in self.feedback)),
                servo_tick=self.count,
                history_contract=HISTORY_CONTRACT,
            )
        except BaseException as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            self.stop()
            raise

    def timing(self):
        elapsed = (
            (self.clock() if self.stopped is None else self.stopped) - self.started
            if self.started is not None
            else 0
        )
        return dict(
            ticks=self.count,
            elapsed_s=elapsed,
            achieved_hz=self.count / elapsed if elapsed > 0 else 0,
            requested_hz=self.config.rate_hz,
            missed_periods=self.misses,
            period_p50_ms=float(np.median(self.periods) * 1000) if self.periods else None,
            period_p99_ms=float(np.percentile(self.periods, 99) * 1000) if self.periods else None,
            computation_and_io_p99_ms=float(np.percentile(self.compute_times, 99) * 1000)
            if self.compute_times
            else None,
            timing_window="last_10000_ticks",
            hardware_rate_validated=False,
        )

    def stop(self):
        if self.stopped is None:
            self.stopped = self.clock()
        self.active = False
        errors = []
        for side, driver in zip(("left", "right"), self.drivers):
            try:
                errors.extend(f"{side}: {error}" for error in driver.disable())
            except Exception as exc:
                errors.append(f"{side}: {type(exc).__name__}: {exc}")
        if errors:
            self.fault = (
                (self.fault or "shutdown")
                + "; motor-off delivery not confirmed: "
                + "; ".join(errors)
            )
        return errors


def run(args):
    """ABC supervisor child: no stock follower, motor server or CAN thread exists."""
    import zmq
    from deploy.robot.config import get_i2rt_config

    profile = get_i2rt_config()
    cfg, setup, dynamics, metadata = prepare_setup(
        args.osc_config,
        profile,
        Path(args.abc_root) / "abc_sim/models/assets/i2rt_yam/yam.xml",
        require_hardware=True,
    )
    drivers, servo, socket = [], None, None
    context = zmq.Context()
    retries = getattr(args, "eef_rejection_retries", 0)
    log = dict(
        contract=metadata,
        rejection_retries=retries,
        status="starting",
        captures=[],
        accepted_targets=[],
        rejected_targets=[],
        telemetry=[],
        fault=None,
    )
    pending = False
    try:
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 10)
        socket.bind(endpoint(args.ipc_dir))
        # Wait for policy/camera setup before opening CAN or enabling any motor.
        # This finite startup wait is unpowered; target TTL starts after enable.
        log["status"] = "waiting_for_reset"
        if not socket.poll(round(1000 * cfg.startup_timeout)):
            raise OscFault("OSC startup timed out waiting for the first reset; CAN was not opened.")
        request = socket.recv_json()
        pending = True
        if request.get("operation") == "shutdown":
            socket.send_json(dict(ok=True))
            pending = False
            log["status"] = "complete"
            return
        if request.get("operation") != "reset":
            raise OscFault("OSC requires reset as its first request before motor enable.")
        # SIGINT/exception cleanup covers partial startup and both CAN interfaces.
        for side in ("left", "right"):
            interface = open_pinned_can(profile.robots[side].follower.channel, cfg.can_timeout)
            drivers.append(SynchronousYamDriver(interface, setup["arms"][side], cfg))
        controllers = [
            ArmOsc(dynamics[i], cfg, setup["arms"][side]["gripper_limits"])
            for i, side in enumerate(("left", "right"))
        ]
        servo = OscServo(controllers, drivers, cfg, rejection_retries=retries)
        servo.start()
        snapshot = servo.tick()
        log["captures"].append(snapshot)
        socket.send_json(dict(ok=True, **snapshot, timing=servo.timing()))
        pending = False
        log["status"] = "running"
        time.sleep(max(0.0, servo.last_tick + 1 / cfg.rate_hz - time.monotonic()))
        while True:
            started = time.monotonic()
            servo.check_health(started)
            request = None
            rejection = None
            if socket.poll(0):
                request = socket.recv_json()
                pending = True
                received = time.monotonic()
                operation = request.get("operation")
                if operation == "shutdown":
                    errors = servo.stop()
                    socket.send_json(dict(ok=not errors, error=servo.fault, timing=servo.timing()))
                    pending = False
                    if errors:
                        raise OscFault(servo.fault)
                    break
                if operation == "reset":
                    servo.reset_target(received)
                elif operation == "target":
                    try:
                        servo.submit(
                            request["action"], request["sequence"], request["sent_at"], received
                        )
                    except EefTargetRejected as exc:
                        rejection = str(exc)
                        log["rejected_targets"].append(
                            dict(
                                request=request,
                                error=rejection,
                                retryable=True,
                                consecutive_rejections=servo.consecutive_rejections,
                                retained_target=servo.target.tolist(),
                                retained_sequence=servo.sequence,
                            )
                        )
                    except OscFault as exc:
                        if isinstance(exc.__cause__, EefTargetRejected):
                            log["rejected_targets"].append(
                                dict(
                                    request=request,
                                    error=str(exc),
                                    retryable=False,
                                    consecutive_rejections=servo.consecutive_rejections,
                                )
                            )
                        raise
                elif operation != "state":
                    raise OscFault(f"Unknown OSC operation: {operation!r}.")
            snapshot = servo.tick()
            if request is not None and request["operation"] == "reset":
                log["captures"].append(snapshot)
            if request is not None and request["operation"] == "target":
                if rejection is None:
                    log["accepted_targets"].append(dict(request=request, acknowledgement=snapshot))
            if servo.count % max(1, round(cfg.rate_hz / 10)) == 0:
                log["telemetry"].append(snapshot)
            if pending:
                reply = dict(ok=rejection is None, **snapshot, timing=servo.timing())
                if rejection is not None:
                    reply.update(
                        retryable=True, error=rejection, rejected_sequence=request["sequence"]
                    )
                socket.send_json(reply)
                pending = False
            time.sleep(max(0.0, started + 1 / cfg.rate_hz - time.monotonic()))
        log["status"] = "complete"
    except BaseException as exc:
        log.update(status="failed", fault=f"{type(exc).__name__}: {exc}")
        if servo is not None:
            servo.fault = servo.fault or log["fault"]
        raise
    finally:
        errors = servo.stop() if servo is not None else [e for d in drivers for e in d.disable()]
        if servo is not None:
            log.update(timing=servo.timing(), fault=servo.fault or log["fault"])
        if errors:
            log.update(status="failed", shutdown_errors=errors)
        if pending and socket is not None:
            try:
                socket.send_json(dict(ok=False, error=log["fault"]))
            except Exception as exc:
                log["error_reply_failure"] = str(exc)
        if socket is not None:
            socket.close()
        context.term()
        if Path(args.output_dir).is_dir():
            write_json(Path(args.output_dir) / "osc_servo.json", log)
        else:
            print(
                f"OSC ended before output initialization: {log['status']}; fault={log['fault']}",
                flush=True,
            )
