import numpy as np
from pynput import keyboard

class ActionsPolicy:
    def __init__(self, comp_actions):
        self.actions_v = comp_actions[:-1]
        self.actions_p = comp_actions[-1]

    def get_actions_p(self):
        return self.actions_p

    def get_action_v(self, i):
        return self.actions_v[i]

class KeyboardPolicy:
    def __init__(self, init_p, v_lin=0.003, v_ang=0.03):
        self.actions_p = init_p
        self.keys_activated = set()
        self.linear_v_mag = v_lin
        self.angular_v_mag = v_ang

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.keys_activated.add(key.char)
        except:
            pass

    def on_release(self, key):
        try:
            self.keys_activated.remove(key.char)
        except:
            pass

    def get_actions_p(self):
        return self.actions_p


class KeyboardPolicy_vxy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v

class KeyboardPolicy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v

class KeyboardPolicy_vxy(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(3)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        return action_v