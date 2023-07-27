import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

 # 모델 로드
json_file = open('models/multi_class_m.json', 'rb')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/multi_class_ml.h5")
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#loss function 확인 필요

model = loaded_model

# 입력된 밸리데이션 set으로 모델을 평가하는 함수
def evaluate_model(validation_file_path):
    # 밸리데이션 set을 csv 파일로 불러오기
    validation_data = pd.read_csv(validation_file_path)

    #데이터 전처리
    validation_data.drop(labels=['TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1, inplace=True)

    #scaler 추가해야함

    # 타겟 컬럼 분리
    validation_target = validation_data[['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']].values
    validation_features = validation_data.drop(labels=['Pastry','Z_Scratch','K_Scatch',	'Stains',	'Dirtiness','Bumps','Other_Faults'],axis=1).values

    # validation_set으로 모델 평가
    predictions = model.predict(validation_features)

    # 모델 평가 결과 계산
    loss = mean_squared_error(validation_target, predictions)
    accuracy = accuracy_score(validation_target.argmax(axis=1), predictions.argmax(axis=1))

    # loss 값과 accuracy 값을 반환
    return loss, accuracy