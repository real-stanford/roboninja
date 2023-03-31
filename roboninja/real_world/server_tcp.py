import pickle
import socket
import time

import RPi.GPIO as GPIO
from hx711 import HX711

IP = '0.0.0.0'
PORT = 23333
f = 80
dt =  1.0 / f

hx = HX711(23, 24)
hx.set_reading_format("MSB", "MSB")

ignore_values = [-1]
for i in range(6):
    ignore_values.append((524288 << i) - 1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((IP, PORT))
    s.listen()

    c, addr = s.accept()
    with c:
        print(addr, 'connected')
        start_time = time.perf_counter()
        idx = 0
        while True:
            idx += 1
            # time.sleep(max(0, idx * dt - (time.perf_counter() - start_time)))
            try:
                val = int(hx.read_long())
                if val in ignore_values:
                    print('ignore', val)
                    continue
                data = pickle.dumps({'idx': idx, 'val': val})
                # if val > 600000:
                #     print(val)
                c.sendall(data)
            except:
                print('disconnected')
                GPIO.cleanup()
                break