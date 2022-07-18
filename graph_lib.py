import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets

df = pd.read_csv(r'C:\Users\ayush\Downloads\openface\OpenFace_2.2.0_win_x64\processed\Lauren_Engagement.csv')
al = pd.read_csv(r'C:\Users\ayush\OneDrive\Documents\Ayushi_Lauren_CSV.csv')

AU = {
    ' AU01_r': "Brow Raiser",
    ' AU04_r': "Brow Lowerer",
    ' AU05_r': "Upper Lid Raiser",
    ' AU06_r': "Cheek Raiser",
    ' AU07_r':  "Lid Tightener",
    ' AU09_r': "Nose Wrinkler",
    ' AU10_r': "Upper Lip Raiser",
    ' AU12_r': "Lip Corner Puller",
    ' AU14_r': "Dimpler",
    ' AU15_r': "Lip Corner Depressor",
    ' AU17_r': "Chin Raiser",
    ' AU20_r': "Lip Stretcher",
    ' AU23_r': "Lip Tightener",
    ' AU25_r': "Lips Part",
    ' AU26_r': "Jaw Drop"
}
d_swap = {v: k for k, v in AU.items()}

def intensity_fig(facial_aciton)