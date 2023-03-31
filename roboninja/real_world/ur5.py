import enum
import multiprocessing as mp
import os
import threading
import time
from queue import Empty
import socket
import pickle

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import threadpoolctl
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class Command(enum.Enum):
    STOP = 0
    CONTINUE = 1
    SERVOL = 2
    START = 3

class Response(enum.Enum):
    COLLISION = 0
    REACH = 1
    SENSOR = 2

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

def udp_worker(ip, port, threshold, sensor_on_event, sensor_kill_event, robot_stop_event, output_queue, verbose):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as c:
        c.sendto(b'start', (ip, port))
        c.settimeout(1)
        last_idx = -1
        while True:
            if sensor_kill_event.is_set():
                if verbose:
                    print('[Sensor] Terminated')
                break
            try:
                data, server = c.recvfrom(1024)
            except:
                robot_stop_event.set()
                print('Sensor connection is broken!')
                raise TimeoutError('Sensor connection is broken!')
            if server[0] == ip and server[1] == port and sensor_on_event.is_set():
                data = pickle.loads(data)
                
                if data['idx'] > last_idx:
                    output_queue.put((Response.SENSOR, time.perf_counter(), data['val']))
                    value = data['val']
                    if value > threshold:
                        if verbose:
                            print(f'[Sensor] Stop. value: {value}, threshold: {threshold}')
                        robot_stop_event.set()
                        sensor_on_event.clear()

class PoseInterpolator:
    def __init__(self, 
            start_pose, end_pose, 
            duration, start_time=0., 
            extrapolate=False,
            reach_target=False):
        assert(duration > 0)
        start_pose = np.array(start_pose)
        end_pose = np.array(end_pose)
        start_pos = start_pose[:3]
        end_pos = end_pose[:3]
        start_rot = st.Rotation.from_rotvec(start_pose[3:])
        end_rot = st.Rotation.from_rotvec(end_pose[3:])
        end_time = start_time + duration
        pos_interp = si.interp1d(
            [start_time, end_time], 
            [start_pos, end_pos], 
            axis=0,
            fill_value='extrapolate')
        rot_interp = st.Slerp(
            [start_time, end_time], 
            st.Rotation.concatenate([start_rot, end_rot]))
        self.pos_interp = pos_interp
        self.rot_interp = rot_interp
        self.extrapolate = extrapolate
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.reach_target = reach_target

    @classmethod
    def create_with_speed_limit(cls, 
            start_pose, end_pose, duration, 
            max_pos_speed, max_rot_speed,
            **kwargs):
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        pos_dist, rot_dist = pose_distance(start_pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        return cls(
            start_pose=start_pose, 
            end_pose=end_pose, 
            duration=duration, **kwargs)

    def __call__(self, t: float):
        first_reach_target = (not self.reach_target) and t > self.end_time
        self.reach_target = t > self.end_time

        t = max(min(t, self.end_time), self.start_time)
        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        pose = np.zeros(6)
        pose[:3] = pos
        pose[3:] = rot.as_rotvec()
        return pose, first_reach_target

class RTDEInterpolationController(mp.Process):
    def __init__(self,
        robot_ip,
        sensor_cfg,
        frequency=125.0,
        lookahead_time=0.1, 
        gain=300,
        max_pos_speed=0.25, # 5% of max speed
        max_rot_speed=0.16, # 5% of max speed
        launch_timeout=1,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=1.05,
        soft_real_time=False,
        verbose=False,
    ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robbin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.sensor_cfg = sensor_cfg


        self.ready_event = mp.Event()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.rtde_r = None

        self.sensor_on_event = mp.Event()

    # ========= launch method ===========
    def start(self):
        self.rtde_r = RTDEReceiveInterface(hostname=self.robot_ip)
        super().start()
        time.sleep(self.launch_timeout)
        assert self.is_alive()
        # if self.verbose:
        #     print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")
    
    def stop(self):
        message = (Command.STOP, None)
        self.input_queue.put(message)
        self.rtde_r.disconnect()
        self.join()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1 / self.frequency))
        pose = list(pose)
        assert len(pose) == 6
        data = (pose, duration)
        message = (Command.SERVOL, data)
        self.input_queue.put(message)
    
    def start_robot(self):
        assert self.is_alive()
        message = (Command.START, None)
        self.input_queue.put(message)
        self.ready_event.wait()

    def sensor_on(self):
        self.sensor_on_event.set()
    
    def sensor_off(self):
        self.sensor_on_event.clear()

    # ========= receive methods =============
    def get_latest_pose(self):
        pose = np.array(self.rtde_r.getActualTCPPose())
        return pose
    
    def get_target_pose(self):
        pose = np.array(self.rtde_r.getTargetTCPPose())
        return pose

    # ========= main loop in process ============
    def run(self):
        # udp client
        sensor_kill_event = threading.Event()
        robot_stop_event = threading.Event()
        self.sensor_thread = threading.Thread(target=udp_worker, args=(
            self.sensor_cfg.ip,
            self.sensor_cfg.port,
            self.sensor_cfg.threshold,
            self.sensor_on_event,
            sensor_kill_event,
            robot_stop_event,
            self.output_queue,
            self.sensor_cfg.verbose
        ), daemon=True)
        self.sensor_thread.start()

        # we only need a single thread for the controller
        limits = threadpoolctl.threadpool_limits(1)

        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)

        try:
            if self.verbose:
                print(f'[RTDEPositionalController] Connect to robot: {robot_ip}')

            # set parameters
            if self.tcp_offset_pose is not None:
                rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency
            robot_status = False
            pose_interpolator = None
            
            t_init = time.perf_counter()
            iter_idx = 0
            while True:
                t_start = time.perf_counter()
                # fetch command from queue
                try:
                    signal, data = self.input_queue.get_nowait()
                except Empty:
                    signal, data = Command.CONTINUE, None
                
                # parse command
                if signal == Command.STOP:
                    break
                elif signal == Command.START:
                    assert robot_status is False
                    assert pose_interpolator is None
                    robot_status = True
                    curr_pose = rtde_r.getActualTCPPose()
                    pose_interpolator = PoseInterpolator(
                        start_pose=curr_pose, 
                        end_pose=curr_pose, 
                        duration=dt, 
                        start_time=time.perf_counter()-dt,
                        reach_target=True
                    )
                    self.ready_event.set()
                    if self.verbose:
                        print('[RTDEPositionalController] Controller is ready')
                elif signal == Command.CONTINUE:
                    pass
                elif signal == Command.SERVOL:
                    assert robot_status is True
                    # if not robot_status:
                    #     pass
                    assert pose_interpolator is not None

                    t_now =time.perf_counter()
                    curr_target_pose, _ = pose_interpolator(t_now)
                    # since curr_pose always lag behind curr_target_pose
                    # if we start the next interpolation with curr_pose
                    # the command robot receive will have discontinouity 
                    # and cause jittery robot behavior.
                    target_pose = data[0]
                    duration = data[1]
                    del data
                    pose_interpolator = PoseInterpolator.create_with_speed_limit(
                        start_pose=curr_target_pose,
                        end_pose=target_pose,
                        duration=duration,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        start_time=time.perf_counter(),
                        reach_target=False
                    )
                    if self.verbose:
                        print(f'[RTDEPositionalController] SERVOL, New pose target:{target_pose}, duration: {pose_interpolator.duration}')
                else:
                    raise NotImplementedError(f'{signal} is not supported!')

                # check status
                if robot_status:
                    assert pose_interpolator is not None

                    # send command to robot
                    t_now = time.perf_counter()
                    pose_command, first_reach_target = pose_interpolator(t_now)
                    vel = 0.5
                    acc = 0.5
                    assert rtde_c.servoL(
                        pose_command,
                        vel, acc, # dummy, not used by ur5
                        dt, 
                        self.lookahead_time, 
                        self.gain)

                    # check sensor
                    if robot_stop_event.is_set():
                        assert rtde_c.servoStop()
                        assert not self.sensor_on_event.is_set()
                        self.output_queue.put((Response.COLLISION, time.perf_counter(), None))
                        robot_status = False
                        pose_interpolator = None
                        robot_stop_event.clear()
                        if self.verbose:
                            print('[RTDEPositionalController] Receive Stop signal from sensor')
                    elif first_reach_target:
                        self.output_queue.put((Response.REACH, time.perf_counter(), None))
                        if self.verbose:
                            print('[RTDEPositionalController] Reach target')

                # regulate frequency
                t_end = time.perf_counter()
                # print(t_end - t_start)
                t_desired = t_init + (iter_idx+1) * dt
                if t_end < t_desired:
                    time.sleep(t_desired - t_end)

                iter_idx += 1

                # if self.verbose:
                #     print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")
            
            rtde_r.disconnect()
            del rtde_r

        finally:
            # kill sensor threading
            sensor_kill_event.set()

            # manditory cleanup
            # decelerate
            rtde_c.servoStop()

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()

            if self.verbose:
                print(f'[RTDEPositionalController] Disconnected from robot: {robot_ip}')


def test_waypoints():
    waypoints = np.array([
        [ 0.3, 0, 0, 2.218,  -2.218,  0.006],
        [ 0.5, 0, 0, 2.218,  -2.218,  0.006],
        [ 0.5, -0.2, 0, 2.218,  -2.218,  0.006],
        [ 0.3, -0.2, 0, 2.218,  -2.218,  0.006],
        [ 0.3, 0, 0, 2.218,  -2.218,  0.006]
    ])
    time_per_waypoint = 1.0
    with RTDEInterpolationController(robot_ip='192.168.0.139') as controller:
        for i, p in enumerate(waypoints):
            controller.servoL(p, duration=time_per_waypoint)
            # if i == 1:
            #     time.sleep(0.5)
            #     controller.servoStop()
            #     break
            time.sleep(time_per_waypoint)
        time.sleep(3)

if __name__ == "__main__":
    test_waypoints()