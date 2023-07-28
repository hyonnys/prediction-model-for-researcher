from fastapi import FastAPI # pip install fast uvicorn
from os import environ as env

"""
<Terminal code>
1. pip install fast uvicorn
2. pip freeze > requirements.txt
3. docker 관련 파일 생성 (.dockerignore, .env, .docker-compose.yaml, Dockerfile)
4. docker와 fastapi 동기화
참고 영상: https://www.youtube.com/watch?v=CzAyaSolZjY
"""

app = FastAPI()

@app.get("/")
def index():
    return f"""Hello, world!!! Secret = {env['MY_VARIABLE']}"""