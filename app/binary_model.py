import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam , RMSprop
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import *
import joblib

# 모델 로드
xgboost_model = joblib.load("models/binary_model.pkl")

# 입력된 밸리데이션 set으로 모델을 평가하는 함수
def evaluate_model(validation_file_path, target_class=1):
    # 밸리데이션 set을 csv 파일로 불러오기
    validation_data = pd.read_csv(validation_file_path)

    #scaler 추가해야함

    # target_class 컬럼 분리
    validation_target = validation_data["target_class"].values
    validation_features = validation_data.drop(columns=["target_class"]).values

    # validation_set으로 모델 평가
    predictions = xgboost_model.predict(validation_features)
    report = classification_report(validation_target, predictions, digits=3, output_dict=True)

    # 원하는 클래스(target_class)에 대한 결과 추출
    target_metrics = report[str(target_class)]

    # F1-score, 정확도(accuracy), 재현율(recall) 값 추출
    f1_score = target_metrics["f1-score"]
    recall = target_metrics["recall"]

    return f1_score, recall