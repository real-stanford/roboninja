import pickle
import socket
import time

import RPi.GPIO as GPIO
from hx711 import HX711
import threading

IP = '0.0.0.0'
PORT = 23333
f = 80
dt =  1.0 / f

hx = HX711(23, 24)
hx.set_reading_format("MSB", "MSB")

ignore_values = [-1, 655359, 786431]
for i in range(6):
    ignore_values.append((524288 << i) - 1)

idx = 0
client_address = None
start_time = 0

def listen(s):
    global client_address
    global start_time
    while True:
        message, address = s.recvfrom(1024)
        print(f'got maeeage from {address}, message = {message}')
        if message == b'start':
            client_address = address
            start_time = time.perf_counter()
        elif message == b'stop':
            client_address = None
        else:
            print('message = ', message)

        print("client_address = ", client_address)


with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('', PORT))
    
    threading.Thread(target=listen, args=(s,), daemon=True).start()
    
    while True:
        if client_address is None:
            time.sleep(0.1)
            continue

        idx += 1
        # time.sleep(max(0, idx * dt - (time.perf_counter() - start_time)))
        try:
            val = int(hx.read_long())
            if val in ignore_values:
                continue
            # if val > 600000:
            #     print(val)
            data = pickle.dumps({'idx': idx, 'val': val})
            s.sendto(data, client_address)
        except:
            print('disconnected')
            GPIO.cleanup()
            break
