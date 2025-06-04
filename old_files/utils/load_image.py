from PIL import Image
from base64 import b64decode
from io import BytesIO
import cv2
from numpy import array
import os


# file path -> PIL.Image
def load_from_file(path:str):
    """file path -> PIL.Image"""
    if not path or not os.path.exists(path):
        return None

    if os.path.getsize(path) == 0:
        return None    
    return Image.open(path)


# base64 -> PIL.Image
def load_from_base64(base64_image_str:str) -> Image:
    """base64 -> PIL.Image"""
    img_data = b64decode(base64_image_str)
    pil_image = Image.open(BytesIO(img_data)).convert("RGB")
    return pil_image


# numpy array -> PIL
def load_from_np_array(np_array: array) -> Image:
    """numpy array -> PIL"""
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image
