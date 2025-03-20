##
## EPITECH PROJECT, 2025
## racingSimulator
## File description:
## agent
##

import socket
import keyboard
import time

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = ('0.0.0.0', 8085)
    s.connect(address)
    return s

def send(s, msg):
    s.sendall(msg.encode('utf-8'))
    response = s.recv(1024).decode('utf-8')
    return response

def loop():
    s = connect()
    speed = 0
    steering = 0

    while True:

        if keyboard.is_pressed(' '):
            break

        if keyboard.is_pressed('z') == keyboard.is_pressed('s'):
            speed = 0

        if keyboard.is_pressed('d') == keyboard.is_pressed('q'):
            steering = 0

        if keyboard.is_pressed('z'):
            speed = min(speed + 0.1, 1)

        if keyboard.is_pressed('s'):
            speed = max(speed - 0.1, -1)

        if keyboard.is_pressed('d'):
            steering = min(steering + 0.1, 0.3)

        if keyboard.is_pressed('q'):
            steering = max(steering - 0.1, -0.3)

        send(s, f'SET_SPEED:{speed}')
        send(s, f'SET_STEERING:{steering}')

        print("\nlidar: ", send(s, 'GET_INFOS_RAYCAST'))

        time.sleep(0.05)

    s.close()

loop()
