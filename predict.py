from ultralytics import YOLO

model = YOLO("yolov8m-seg-custom.pt")
model.predict(source="2.jpg", show = True, save = True, hide_conf = False, conf=0.5, save_txt=False, save_crop=False,  line_thickness = 2)

