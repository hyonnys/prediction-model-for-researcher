from fastapi import FastAPI
from os import environ as env

app = FastAPI()

@app.get("/")
def index():
    return f"""Hello, world!!! Secret = {env['MY_VARIABLE']}"""