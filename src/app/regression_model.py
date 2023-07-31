import pandas as pd
import numpy as np
import joblib
from keras.optimizers import RMSprop
from sklearn.metrics import *
from tensorflow.keras.models import model_from_json
import time

 # 모델 로드
json_file = open('models/regression_m.json', 'rb')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/regression_ml.h5")
loaded_model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.001))

model = loaded_model

# 입력된 밸리데이션 set으로 모델을 평가하는 함수
def evaluate_model(validation_file_path):
    # 밸리데이션 set을 csv 파일로 불러오기
    validation_data = pd.read_csv(validation_file_path)

    #데이터 전처리
    validation_data.drop('Sex', axis=1, inplace=True)

    # Scaler 객체를 불러오기
    scaler = joblib.load('models/regression_scaler.pkl')

    # Rings 타겟 컬럼 분리
    validation_target = validation_data["Rings"].values
    validation_features = validation_data.drop(columns=["Rings"]).values

    # transform
    validation_features = scaler.transform(validation_features)

    #start time
    start_time = time.time()
    # validation_set으로 모델 평가
    predictions = model.predict(validation_features).flatten()
    end_time = time.time()
    mse = mean_squared_error(validation_target, predictions)

    Time_taken = end_time - start_time
    return mse, Time_taken

