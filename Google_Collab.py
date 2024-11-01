from google.colab import drive

drive.mount('/content/gdrive')

ROOT_DIR = '/content/gdrive/My Drive/PRO'

!pip install ultralytics
!pip install torch torchvision

import os
import torch

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=10, batch=20)

import locale
locale.getpreferredencoding = lambda: "UTF-8"
!scp -r /content/runs '/content/gdrive/My Drive/PRO'

%cd /content/gdrive/MyDrive/PRO

!yolo task=detect mode=predict model=runs/detect/train20/weights/best.pt conf=0.80 source=data/images/test
