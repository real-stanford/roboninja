import pickle
import socket
import time


def tcp_worker(q, time_range):
    IP = '192.168.0.174'
    PORT = 23333
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((IP, PORT))
        last_time = time.perf_counter()
        while True:
            data = s.recv(1024)
            data = pickle.loads(data)
            idx = data['idx']
            val = data['val']
            cur_time = time.perf_counter()
            q.append((idx, val, cur_time, 1.0/(cur_time - last_time)))
            last_time = cur_time
            
            while cur_time - q[0][2] > time_range:
                q.popleft()

