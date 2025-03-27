##
## EPITECH PROJECT, 2025
## racingSimulator
## File description:
## agent
##

import socket
import keyboard
import time
import pandas as pd
from datetime import datetime

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
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    file_name = f'LidarData_{timestamp}.csv'

    while True:

        if keyboard.is_pressed(' '):
            break

        if keyboard.is_pressed('z') == keyboard.is_pressed('s'):
            speed = 0

        if keyboard.is_pressed('d') == keyboard.is_pressed('q'):
            steering = 0

        if keyboard.is_pressed('z'):
            speed = min(speed + 0.1, 1)
            print("J'appuie", speed)

        if keyboard.is_pressed('s'):
            speed = max(speed - 0.1, -1)

        if keyboard.is_pressed('d'):
            steering = min(steering + 0.1, 0.3)

        if keyboard.is_pressed('q'):
            steering = max(steering - 0.1, -0.3)

        send(s, f'SET_SPEED:{speed}')
        send(s, f'SET_STEERING:{steering}')

        lidarData = send(s, 'GET_INFOS_RAYCAST')
        lidarSplitData = lidarData.split(':')
        positionData = send(s, 'GET_POSITION')
        print("position", positionData)
        positionSplitData = positionData.split(':')
        speedData = send(s, 'GET_SPEED')
        print("speed", speedData)
        speedSplitData = speedData.split(':')
        if (lidarSplitData[0] != 'OK' and positionSplitData[0] != 'OK' and speedSplitData[0] != 'OK'):
            continue
        df = pd.DataFrame([lidarSplitData[2:]], columns=[f"Ray_{i}" for i in range(len(lidarSplitData[2:]))])
        df['Position_Y'] = positionSplitData[2]
        df['Position_X'] = positionSplitData[3]
        df['Position_Z'] = positionSplitData[4]
        df['Speed'] = speedSplitData[2]
        df.to_csv(file_name, mode='a', header=not pd.io.common.file_exists(file_name), index=False)
        time.sleep(0.05)

    s.close()

loop()
