import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import time
import os
import glob



def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    rows = []
    for line in lines:
        row = line.strip().split(',')
        
        if len(row) == 67:  # Проверяем, чтобы было ровно 67 значений
            rows.append([float(x) if x not in ['открытая', 'закрытая'] else x for x in row])
            
    return rows


inp=read_data('data.csv')
a=inp[0]
a.pop()

print(a)