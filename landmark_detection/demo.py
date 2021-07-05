import cv2

from pathlib import Path
from landmark_detector import LandmarkDetector

model_path = Path('../checkpoints/checkpoint_49.pth.tar')
detector = LandmarkDetector(model_path, device='cpu')

img = cv2.imread('../test_1.jpg')

d = {
    'image': img,
    'box': [819.27, 432.15, 971.70, 575.87]
}

r = detector.predict(d)
print(r)
