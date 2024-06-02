from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train22/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model(r"C:\Users\yfs\Desktop\ultralytics-main\datasets\wave\train\images\6.png",save=True)  # predict on an image
