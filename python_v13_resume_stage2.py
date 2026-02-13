from ultralytics import YOLO
model = YOLO('hoban_v13_stage2/weights/last.pt')
model.train(resume=True)
