import numpy as np
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random
import plotly.graph_objects as go
from xgboost import XGBClassifier



def hello():
    print(f"")


df = pd.read_csv('Lauren_Engagement.csv')
df.columns = df.columns.str.replace(' ', '')
ll = pd.read_csv('Lauren_Lauren_CSV.csv')
al = pd.read_csv('Ayushi_Lauren_CSV.csv')

x_name = np.array([i for i in df.columns.values if i.startswith('x')])
y_name = np.array([i for i in df.columns.values if i.startswith('y')])

start_time = 0
endtime = 389

eyebrow = 0
eye = 0
jaw = 0
innerlip = 0
nose = 0
outerlip = 0
allFace = 0
eyegaze = 0



facial_action = np.array([i for i in df.columns.values if i.endswith('_r')])
AU = ["Upper Brow Raiser", "Lower Brow Raiser", "Brow Lowerer", "Upper Lid Raiser", "Cheek Raiser", "Lid Tightener", "Nose Wrinkler",
      "Upper Lip Raiser", "Lip Corner Puller", "Dimpler", "Lip Corner Depressor", "Chin Raiser", "Lip Stretcher",
      "Lip Tightener", "Lips Part", "Jaw Drop"]
intensity = [0 for i in range(len(AU))]

coor_dir = {"jaw": [0, 16],
            "eyebrow": [17, 27],
            "nose": [27, 36],
            "eye": [36, 48],
            "outerlip": [48, 60],
            "innerlip": [60, 68]}
# Run if Engagement Bar needed
al_arr = []
ll_arr = []
count1 = 0
count2 = 0
for i in range(0, 389 * 25):
    if i / 25 < al.at[count1, 'Column6']:
        if al.at[count1, 'Column9'] == 'E':
            al_arr.append(1)
        else:
            al_arr.append(0)
    else:
        count1 += 1
        if al.at[count1, 'Column9'] == 'E':
            al_arr.append(1)
        else:
            al_arr.append(0)
    if i / 25 < ll.at[count2, 'Column6']:
        if ll.at[count2, 'Column9'] == 'e':
            ll_arr.append(1)
        else:
            ll_arr.append(0)
    else:
        count2 += 1
        if ll.at[count2, 'Column9'] == 'e':
            ll_arr.append(1)
        else:
            ll_arr.append(0)



def figure():
    fig = go.Figure()
    engaged = []
    disengaged = []
    wrong = []
    for i in range(0, len(al_arr), 25):
        if al_arr[i] == 1:
            engaged.append(int(i / 25))
        else:
            disengaged.append(int(i / 25))
    fig.add_trace(go.Scatter(
        x=engaged, y=[0 for i in range(len(engaged))], mode='markers', marker_size=10, name="Engaged"))
    fig.add_trace(go.Scatter(
        x=disengaged, y=[0 for i in range(len(disengaged))], mode='markers', marker_size=10,name="Disengaged"
    ))
    fig.add_trace(go.Scatter(
        x=wrong, y=[0 for i in range(len(wrong))], mode='markers', marker_size=10, name="Model Was WRONG"
    ))
    fig.update_xaxes(showgrid=False, title_text="Seconds")
    fig.update_yaxes(showgrid=False,
                     zeroline=True, zerolinecolor='black', zerolinewidth=3,
                     showticklabels=False,)
    fig.update_layout(height=200, plot_bgcolor='white')
    fig.show()

def wrong_figure(y_pred):
    fig = go.Figure()
    wrong = []
    right = []
    for i in range(0, len(al_arr), 25):
        if al_arr[i] != y_pred[i]:
            wrong.append(int(i / 25))
        else:
            right.append(int(i/25))
    fig.add_trace(go.Scatter(
        x=wrong, y=[0 for i in range(len(wrong))], mode='markers', marker_size=10, name="Model Was WRONG"
    ))
    fig.add_trace(go.Scatter(
        x=right, y=[0 for i in range(len(right))], mode='markers', marker_size=5, name="Model Was RIGHT"
    ))
    fig.update_xaxes(showgrid=False, title_text="Seconds")
    fig.update_yaxes(showgrid=False,
                     zeroline=True, zerolinecolor='black', zerolinewidth=3,
                     showticklabels=False, )
    fig.update_layout(height=200, plot_bgcolor='white')
    fig.show()



class Bar:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)

    def update_frame(self, value, confidence):
        if value == 1:
            new_graph = np.zeros((self.height, self.width, 3), np.uint8)
            new_graph[-5:, :, :-2] = 255
            self.graph = new_graph
        else:
            new_graph = np.zeros((self.height, self.width, 3), np.uint8)
            new_graph[-1*int((self.height*confidence)):, :, :-2] = 255
            self.graph = new_graph

    def get_graph(self):
        return self.graph

# def kappa():
#     print("Cohen's Kappa:", cohen_kappa_score(al_arr, ll_arr))

#Run if graph needed
class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)
    def update_frame(self, value, engage, wrong):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        if engage == 0:
            new_graph[:,-1,:-2] = 255
        if wrong == 1:
            new_graph[:, -1] = (0,0,255)
        new_graph[self.height - value:self.height - value + 3,-1,:] = 255
        new_graph[self.height - value:,-1,:] = 255
        self.graph = new_graph
    def get_graph(self):
        return self.graph

pastTime = 0

global y_pred
y_pred = []







eye_1 = np.array([i for i in df.columns.values if i.startswith('gaze_angle')])
eye_lmk_20_x = np.array([i for i in df.columns.values if i.startswith('eye_lmk_x_20')])
eye_lmk_20_y = np.array([i for i in df.columns.values if i.startswith('eye_lmk_y_20')])
eye_lmk_50_x = np.array([i for i in df.columns.values if i.startswith('eye_lmk_x_50')])
eye_lmk_50_y = np.array([i for i in df.columns.values if i.startswith('eye_lmk_y_50')])

def visual(y_pred):
    # Run to show visualization
    cap = cv2.VideoCapture('Lauren_Engagement.mp4')  # opening video
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * start_time)  # setting start time
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
    graph = Graph(int(width), 60)  # initializing graph
    bar = Bar(int(width / 15), int(height / 2))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        time.sleep(1/fps) #to slow down the video
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(width), int(height)))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        currtime = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        x_coor = df.loc[count, x_name].values / 2  # feature x and y coordinates
        y_coor = df.loc[count, y_name].values / 2
        if allFace == 1:
            for i in range(0, len(x_coor)):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)  # drawing coordinates on video
        if eyebrow == 1:
            for i in range(coor_dir["eyebrow"][0], coor_dir["eyebrow"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if eye == 1:
            for i in range(coor_dir["eye"][0], coor_dir["eye"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if jaw == 1:
            for i in range(coor_dir["jaw"][0], coor_dir["jaw"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if nose == 1:
            for i in range(coor_dir["nose"][0], coor_dir["nose"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if innerlip == 1:
            for i in range(coor_dir["innerlip"][0], coor_dir["innerlip"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if outerlip == 1:
            for i in range(coor_dir["outerlip"][0], coor_dir["outerlip"][1]):
                cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (0, 0, 0), 1)
        if eyegaze == 1:
            eye_1_coor = df.loc[count, eye_1].values * 57.2957795  # feature x and y coordinates
            center_0_x = df.loc[count, eye_lmk_20_x].values / 2  # feature x and y coordinates
            center_0_y = df.loc[count, eye_lmk_20_y].values / 2
            center_1_x = df.loc[count, eye_lmk_50_x].values / 2  # feature x and y coordinates
            center_1_y = df.loc[count, eye_lmk_50_y].values / 2
            cv2.line(frame, (int(center_1_x[0]), int(center_1_y[0])),
                     (int(center_1_x[0] + eye_1_coor[0]), int(center_1_y[0] + eye_1_coor[1])), (0, 0, 0), 2)
            cv2.line(frame, (int(center_0_x[0]), int(center_0_y[0])),
                     (int(center_0_x[0] + eye_1_coor[0]), int(center_0_y[0] + eye_1_coor[1])), (0, 0, 0), 2)

        if pastTime != 0:
            color = 255 / (pastTime * 2)
            for i in range(1, (pastTime * 2) + 1):
                x_coor = df.loc[count - i, x_name].values / 2  # feature x and y coordinates
                y_coor = df.loc[count - i, y_name].values / 2
                if allFace == 1:
                    for i in range(0, len(x_coor)):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color),
                                   1)  # drawing coordinates on video
                if eyebrow == 1:
                    for i in range(coor_dir["eyebrow"][0], coor_dir["eyebrow"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                if eye == 1:
                    for i in range(coor_dir["eye"][0], coor_dir["eye"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                if jaw == 1:
                    for i in range(coor_dir["jaw"][0], coor_dir["jaw"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                if nose == 1:
                    for i in range(coor_dir["nose"][0], coor_dir["nose"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                if innerlip == 1:
                    for i in range(coor_dir["innerlip"][0], coor_dir["innerlip"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                if outerlip == 1:
                    for i in range(coor_dir["outerlip"][0], coor_dir["outerlip"][1]):
                        cv2.circle(frame, (int(x_coor[i]), int(y_coor[i])), 1, (color, color, color), 1)
                color += 255 / (pastTime * 2)
        facial = df.loc[count, facial_action].values
        ind = intensity.index(1)
        wrong = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        if y_pred[count] != al_arr[count]:
            wrong = 1
            cv2.putText(frame, 'Model Predicted WRONG', (int(width - 200), int(height / 5 - 5)), font, 0.5, (0, 0, 255), 2)
        graph.update_frame(int(facial[ind]* 10), al_arr[count], wrong)  # update to frame
        roi = frame[-70:-10, -1 * int(width):, :]  # height of graph
        roi[:] = graph.get_graph()  # copy current graph

        string = "X-Axis: Time in seconds, Y-Axis: Intensity of {}".format(AU[ind])
        cv2.putText(frame, string, (int(width / 4), int(height - height / 7)), font, 0.5, (255, 255, 255), 2)
        confidence = df.loc[1, 'confidence']
        bar.update_frame(int(y_pred[count]), confidence)  # update to frame
        roi1 = frame[-405:-135, -74:-10, :]  # height of graph
        roi1[:] = bar.get_graph()  # copy current graph
        cv2.putText(frame, 'Disengaged', (int(width - 96), int(height / 4 - 5)), font, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, 'Engaged', (int(width - 96 + 20), int((height - height / 4) + 20)), font, 0.5, (0, 0, 0),
                    1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if currtime / 1000 >= endtime:
            break  # end when time is up
        if count == 9724:
            break
    cap.release()
    cv2.destroyAllWindows()
