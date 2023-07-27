import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from flask import send_from_directory

app = Flask(__name__)

# 피클 파일로부터 정규화 정보 불러오기
mean_value = 0.5
std_value = 0.25
# with open('scaler.pkl', 'rb') as f:
#     normalization_info = pickle.load(f)
#     mean_value, std_value = normalization_info

# 모델 불러오기
model = tf.keras.models.load_model('model.h5')

# 입력값 전처리 함수
def preprocess_input(input_values):
    # 입력값을 numpy 배열로 변환
    input_values = np.array(input_values, dtype=np.float32)
    # 정규화 적용
    input_values = (input_values - mean_value) / std_value
    # 모델에 맞는 형태로 차원 변경 등의 처리 (예: 배치 차원 추가)
    input_values = np.expand_dims(input_values, axis=0)  # 예시로 입력값이 1차원인 경우 배치 차원 추가
    return input_values

# 모델 실행 함수
def run_model(input_values):
    input_tensor = preprocess_input(input_values)
    output = model.predict(input_tensor)
    return output[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 사용자가 입력한 27개의 값 가져오기
        values = [float(request.form[f'value{i}']) for i in range(1, 28)]
        
        # 모델 실행
        result = run_model(values)
        
        # 결과 반환 (템플릿에 전달하여 출력)
        return render_template('result.html', result=result)
    else:
        return render_template('index.html')
    
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)
    
if __name__ == '__main__':
    app.run(debug=True)
