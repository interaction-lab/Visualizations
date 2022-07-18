import numpy as np
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\ayush\Downloads\openface\OpenFace_2.2.0_win_x64\processed\WIN_20220626_18_50_25_Pro.csv')

def play_camera():
    x_name = np.array([i for i in df.columns.values if i.startswith(' x')])
    y_name = np.array([i for i in df.columns.values if i.startswith(' y')])
    start_time = int(input("Start time in seconds: "))
    endtime = int(input("End time in seconds: "))
    cap = cv2.VideoCapture('singing_vid.mp4')
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000*start_time)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()
    x = np.linspace(0, width, frames)
    y = x / 2. + 100 * np.sin(2. * np.pi * x / 1200)
    while cap.isOpened():
        fig.clf()
        ret, frame = cap.read()
        time.sleep(1/fps)
        count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        currtime = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        x_coor = df.loc[count, x_name].values
        y_coor = df.loc[count, y_name].values
        plt.imshow(frame)
        plt.plot(x, y, 'k-', lw=2)
        plt.plot(x[count-1], y[count-1], 'or')
        plt.pause(0.01)
        for i in range(0, len(x_coor)):
            cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if currtime/1000 >= endtime:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    play_camera()

