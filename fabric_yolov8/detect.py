import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/home/admin_bift/mxt/yolov8_2024/runs/train/exp154/weights/best.pt') # select your model.pt path
    model.predict(source=r'/home/admin_bift/mxt/yolov9-202400414/dataset/detectimg',
                  imgsz=640,
                  project='runs/detect',
                  device='1',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                )



