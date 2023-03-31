import threading
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

from client_tcp import tcp_worker
from client_udp import udp_worker


class Monitor:
    def __init__(self, time_range):
        matplotlib.use('TkAgg')
        plt.ion()
        self.fig = plt.figure()

        self.time_range = time_range
        self.val_range = 2000000
        self.f_range = 200

        self.ax_val = self.fig.add_subplot(211)
        self.line_val, = self.ax_val.plot([0], [0])
        self.ax_val.set_xbound(0, self.time_range)
        self.ax_val.set_ybound(0, self.val_range)
        self.ax_val.set_title('value')


        self.ax_f = self.fig.add_subplot(212)
        self.line_f, = self.ax_f.plot([0], [0])
        self.ax_f.set_xbound(0, self.time_range)
        self.ax_f.set_ybound(0, self.f_range)
        self.ax_f.set_title('frequence')

        self.last_t = time.perf_counter()

    def update(self, q):
        data = list(q)
        x_data = [data[-1][2] - x[2] for x in data]
        val_data = [x[1] for x in data]
        f_data = [x[3] for x in data]
        self.line_val.set_data(x_data, val_data)
        self.line_f.set_data(x_data, f_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    time_range = 3
    q = deque()
    m = Monitor(time_range)
    threading.Thread(target=udp_worker, args=(q, time_range), daemon=True).start()
    while True:
        time.sleep(0.1)
        m.update(q)
        

if __name__=='__main__':
    main()
