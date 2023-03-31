import copy
import os
import pickle
import time
from queue import Empty

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roboninja.env.base_env import BaseEnv
from roboninja.real_world.multi_realsense import MultiRealsense
from roboninja.real_world.ur5 import Response, RTDEInterpolationController
from roboninja.utils.visualizer import Visualizer


class Display:
    def __init__(self):
        self.visualizer = Visualizer()
        self.cur_img = self.visualizer.empty_image(num_dim=3)
        self.cur_sensor_val = 0

    def show(self):
        img = cv2.putText(
            self.cur_img.copy(),
            str(self.cur_sensor_val),
            (160, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=2,
            color=(0, 0 ,255)
        )
        img = img[:, :, slice(None, None, -1)]
        cv2.imshow('display', img)
        cv2.pollKey()

    def update_img(self, img):
        self.cur_img = img.copy()
        self.show()
    
    def update_sensor_val(self, sensor_val):
        self.cur_sensor_val = sensor_val
        self.show()


class RealEnv(BaseEnv):
    def __init__(self,
        controller:RTDEInterpolationController,
        realsense:MultiRealsense,
        offset:np.ndarray,
        duration:float,
        scale:float,
        terminate_height:float,
        global_rot_center:list,
        global_rot_angle:float
    ):
        super().__init__()
        self.controller = controller
        self.display = Display()
        self.realsense = realsense
        self.offset = offset
        self.duration = duration
        self.scale = scale
        self.last_wrd_pos = None
        self.min_height = terminate_height + 0.005
        self.global_rot_center = np.array(global_rot_center)
        self.global_rot_angle = global_rot_angle

        self.visualizer = Visualizer()
        self.sign = 1

    def wrd2ur5(self, wrd_pos):
        print(wrd_pos)
        scale = self.scale
        p = np.array([0, -wrd_pos[0], wrd_pos[1]])


        p[0] = self.sign * 0.01
        self.sign *= -1

        # glb_scale = 0.85
        glb_scale = 1.1
        center = np.array([0, -0.5, 0.0175])
        dp = p - center
        p = dp * glb_scale + center

        p = p * scale + self.offset
        rot = Rotation.from_euler('x', -wrd_pos[2]) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])
        
        rot_angle = self.global_rot_angle
        dp = p[:2] - self.global_rot_center

        p[:2] = self.global_rot_center + np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]]) @ dp
        rot = Rotation.from_euler('z', rot_angle) * rot
        
        r = rot.as_rotvec()
        return list(p) + list(r)

    def _move(self, ur5_pos, duration=None, sensor=False, **kwargs):
        duration = self.duration if duration is None else duration
        while True:
            try:
                self.controller.output_queue.get(False)
            except Empty:
                break
        if sensor: self.controller.sensor_on()
        self.controller.servoL(ur5_pos, duration=duration)
        
        stop_signal = False

        while True:
            signal, t, data = self.controller.output_queue.get()

            if signal == Response.COLLISION:
                stop_signal = True
                break
            elif signal == Response.REACH:
                break
            elif signal == Response.SENSOR:
                self.update_sensor_info(data)
                pass
                
            # assert t - last_timestep > 0
            last_timestep = t
            
        self.controller.sensor_off()

        return stop_signal

    def gen_vis(self):
        state_estimation = self.info['state_estimaiton'][:, :, None]
        gt = self.info['img_bone'][:, :, None]

        mask = state_estimation < 1
        state_estimation = state_estimation * (1 - mask) + mask * np.array([168, 120, 85]) / 255.0
        mask = gt < 1
        gt = gt * (1 - mask) + mask * np.array([120, 120, 120]) / 255.0

        img_vis = state_estimation * 0.6 + gt * 0.4

        for c in self.info['backward_pos_pair_list']:
            img_vis = self.visualizer.vis_pos(
                np.asarray(c)[:, :2], img_vis, 
                color=(0.8, 0.2, 0.2), thickness=2
            )
        
        img_vis = self.visualizer.vis_pos(
            np.asarray(self.info['forward_pos_list'])[:, :2], img_vis,
            color=(0, 0, 0.8), thickness=2
        )
        return img_vis
    
    def move(self, wrd_pos, pre_pos, step_idx=None, roll_back=False, horizontal=False, **kwargs):
        if roll_back:
            self.forward_pos_list.pop()
            self.backward_pos_pair_list.append([copy.deepcopy(wrd_pos), copy.deepcopy(pre_pos)])
        else:
            if not horizontal:
                self.forward_pos_list.append(copy.deepcopy(wrd_pos))

        self.info['forward_pos_list'] = self.forward_pos_list
        self.info['backward_pos_pair_list'] = self.backward_pos_pair_list
        self.info['roll_back'] = roll_back

        self.policy_info_history.append((time.time() - self.start_time, copy.deepcopy(self.info)))
        vis_img = self.gen_vis()

        if self.display is not None:
            self.display.update_img(vis_img)
        self.policy_info_history.append((time.time() - self.start_time, copy.deepcopy(self.info)))
        self.last_wrd_pos = wrd_pos
        return self._move(self.wrd2ur5(wrd_pos), **kwargs)

    def roll_back(self,
        wrd_pos_history:list,
        n_back:int,
        step_idx:int
    ):
        self.controller.start_robot()
        return super().roll_back(wrd_pos_history, n_back, step_idx)
    
    def reset(self, reset_pos, output_dir, **kwargs):
        self.forward_pos_list = [reset_pos]
        self.backward_pos_pair_list = list()
        self.wrd_reset_pos = reset_pos.copy()

        self.controller.start_robot()
        self.reset_pos = self.wrd2ur5(reset_pos)
        reset_duration = 1.5
        # self.reset_pos[0] -= 0.15
        # self._move(ur5_pos=self.reset_pos, duration=reset_duration, sensor=False)
        # self.reset_pos[0] += 0.15
        self._move(ur5_pos=self.reset_pos, duration=reset_duration, sensor=False)
        self.start_time = time.time()
        self.sensor_info_history = list()
        self.policy_info_history = list()
        self.output_dir = output_dir
        video_path = os.path.join(self.output_dir, 'videos')
        if self.realsense is not None:
            self.realsense.start_recording(
                video_path=video_path,
                start_time=self.start_time
            )

    def terminate(self, success, **kwargs):
        wrd_pos = self.last_wrd_pos
        ur5_pos = self.wrd2ur5(wrd_pos)
        if success:
            # move right
            ur5_pos[1] += 0.05
            self._move(ur5_pos, 2, False)

            # reset
            ur5_pos = self.reset_pos.copy()
            self._move(ur5_pos, 1.5, False)

        if self.realsense is not None:
            self.realsense.stop_recording()
        pickle.dump(self.sensor_info_history, open(os.path.join(self.output_dir, 'sensor_info_history.pkl'), 'wb'))
        pickle.dump(self.policy_info_history, open(os.path.join(self.output_dir, 'policy_info_history.pkl'), 'wb'))

    def update_info(self, **kwargs):
        self.info = kwargs

    def update_sensor_info(self, data):
        self.display.update_sensor_val(data)
        self.sensor_info_history.append((time.time() - self.start_time, data))

    @property
    def energy(self):
        return None