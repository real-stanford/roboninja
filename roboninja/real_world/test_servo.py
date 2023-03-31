from time import time
import rtde_control
import time
import threading
from client_udp import udp_worker
from collections import deque
import numpy as np

time_range = 3
q = deque()

rtde_c = rtde_control.RTDEControlInterface("192.168.0.139")

# Parameters
lookahead_time = 0.1
gain = 300
T = 15
dt = 1.0 / 125
N = int(T / dt)
# force_threshold = 550000 # air
# force_threshold = 2800000 # 3d printed knife
force_threshold = 1300000 # real knife



home_p = np.array([0.3, 0, 0, 2.2, -2.2, 0])
target_p = np.array([0.3, 0, -0.15, 2.2, -2.2, 0])
rtde_c.moveL(home_p, 0.05, 0.1)

threading.Thread(target=udp_worker, args=(q, time_range), daemon=True).start()

while True:
    if len(q) > 0:
        print('==> start')
        break
    time.sleep(0.01)

for i in range(N):
    tmp_target = home_p + (target_p - home_p) / N * (i+1)
    rtde_c.servoL(tmp_target, 0.5, 0.5, dt, lookahead_time, gain)
    val = q[-1][1]
    print(i, val)
    if val > force_threshold and i > 500:
        break
    time.sleep(dt)

rtde_c.stopScript()
rtde_c.servoStop()

# rtde_c.moveL(home_p, 0.5, 0.1)