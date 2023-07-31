import pandas as pd
import joblib
import time
from sklearn.metrics import *
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error, accuracy_score

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

    # Scaler 객체를 불러오기
    scaler = joblib.load('models/multi_scaler.pkl')

    # 타겟 컬럼 분리
    validation_target = validation_data[['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']].values
    validation_features = validation_data.drop(labels=['Pastry','Z_Scratch','K_Scatch',	'Stains',	'Dirtiness','Bumps','Other_Faults'],axis=1).values

    # transform
    validation_features = scaler.transform(validation_features)
    
    #start time
    start_time = time.time()

    # validation_set으로 모델 평가
    predictions = model.predict(validation_features)
    end_time = time.time()
    # 모델 평가 결과 계산
    loss = mean_squared_error(validation_target, predictions)
    accuracy = accuracy_score(validation_target.argmax(axis=1), predictions.argmax(axis=1))
    Time_taken = end_time - start_time
    # loss 값과 accuracy 값을 반환
    return loss, accuracy, Time_taken