# StrawberryDetection
the project of detecting strawberry for roommate, using yolov11 model

//download your datasets and put it in the root directory, two folders, "labels"&"images" should be in the dataset folder
下载你的草莓数据集文件，文件名为datasets，放在ultralytics文件夹下，datasets文件夹里要有train的image和label以及test的image和label

//datasets里的data.yaml放在ultralytics文件夹

//writing train.py

from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')
    # model.load('yolov8n.pt')
    model.train(data='./data.yaml', epochs=20, batch=4, device='0', imgsz=640, workers=2, cache=False,
                amp=True, mosaic=False, project='runs/train', name='exp')


//finishing training dataset
