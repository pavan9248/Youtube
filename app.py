import uvicorn
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import pickle
import numpy as np
import os
import requests
import subprocess
import shutil
import easyocr
from yolov5.detect import counter

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
