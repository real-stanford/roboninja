import pickle
import socket
import time


def udp_worker(q, time_range):
    IP = '192.168.0.174'
    PORT = 23333

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as c:
        c.sendto(b'start', (IP, PORT))
        last_time = time.perf_counter()
        last_idx = -1
        while True:
            data, server = c.recvfrom(1024)
            
            if server[0] == IP and server[1] == PORT:
                data = pickle.loads(data)
                if data['idx'] > last_idx:
                    idx = data['idx']
                    val = data['val']
                    cur_time = time.perf_counter()
                    if q is not None:
                        q.append((idx, val, cur_time, 1.0/(cur_time - last_time)))
                        last_idx  = idx
                        last_time = cur_time
                        while cur_time - q[0][2] > time_range:
                            q.popleft()
                    else:
                        print(val)

if __name__=='__main__':
    udp_worker(None, None)

