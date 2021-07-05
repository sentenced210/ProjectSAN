import pytest

import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from landmark_detection.landmark_detector import LandmarkDetector
import cv2
from pathlib import Path


def test_1():
    model_path = Path('../checkpoints/checkpoint_49.pth.tar')
    detector = LandmarkDetector(model_path, device='cpu')
    assert detector.param.argmax_size == 3


def test_2():
    model_path = Path('../checkpoints/checkpoint_49.pth.tar')
    detector = LandmarkDetector(model_path, device='cpu')
    img = cv2.imread('../test_1.jpg')
    d = {
        'image': img,
        'box': [819.27, 432.15, 971.70, 575.87]
    }
    r = detector.predict(d)
    assert r['error'] == ''


def test_3():
    model_path = Path('../checkpoints/checkpoint_49.pth.tar')
    detector = LandmarkDetector(model_path, device='cpu')
    img = cv2.imread('../test_1.jpg')
    d = {
        'image': img,
        'box': [819.27, 432.15, 971.70]
    }
    r = detector.predict(d)
    assert r['error'] != ''
