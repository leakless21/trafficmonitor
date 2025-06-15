import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any

import cv2
import numpy as np
from loguru import logger

from fast_plate_ocr import ONNXPlateRecognizer
from ..utils.custom_types import PlateDetectionMessage, OCRResultMessage
