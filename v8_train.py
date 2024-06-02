# from ultralytics import YOLO
# model = YOLO("/public/home/wangmaofa2008/yfs/v8/ultralytics-main/runs/detect/train31/weights/last.pt")
# model.train(data="datasets/zhenghe/data.yaml", workers=0, epochs=2, batch=32, patience=100, resume=True,)

from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("/public/home/wangmaofa2008/yfs/v8/ultralytics-main/runs/detect/train31/weights/last.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
# model.train(data="datasets/zhenghe/data.yaml", epochs=3,workers=0,batch=32,patience=100,resume=True,)  # 训练模型
model.train(data="datasets/zhenghe/data.yaml", workers=8, epochs=2000, batch=32, patience=55, device=0, )