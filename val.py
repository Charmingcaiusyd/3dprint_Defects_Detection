import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/public/home/wangmaofa2008/yfs/v8/ultralytics-main/runs/detect/train29/weights/best.pt')
    model.val(data='/public/home/wangmaofa2008/yfs/v8/ultralytics-main/datasets/zhenghe/data.yaml',
           
              project='runs/val',
              name='exp',
              )