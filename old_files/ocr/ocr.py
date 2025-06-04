import numpy as np
import easyocr

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from io import BytesIO

import warnings
# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def boxes_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check if the boxes overlap
    if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
        return True
    return False

def bounding_box(box):
    return [box[0][0],
            min(box[0][1],box[2][1]),
            box[2][0],
            max(box[0][1],box[2][1])]

class Reader():
    def __init__(self):
        """
            Create reader based on EasyOCR
        """
        self.translation_table = {
                'a':'а',  'b':'ь',  'c':'с',
                'e':'е',  'o':'о',  'p':'р',
                'x':'х',  'y':'у',  '^':'л',
                'A':'А',  'B':'В',  'C':'С', 
                'E':'Е',  'H':'Н',  'K':'К',
                'M':'М',  'O':'О',  'P':'Р',
                'T':'Т',  'X':'Х',  'Y':'У'
            }
        self.exceptions = [
                "огрн","ао", "гк", "жк"
                "ул","стр","помещ",
                "млн",
                "рф",
                "гоголь"]
        self.eng_check = r'^[A-Za-z]+$'
        self.rus_check = r'^[\u0400-\u04FF]+$'
        self.OCR_Reader = easyocr.Reader(['en','ru'], gpu=True)
    
    def rawtext(self, img):
        """
            Extract text from an image
        """
        if type(img) != np.ndarray:
            img = np.array(self.imagepreproc(img))
        
        picture_results = self.OCR_Reader.readtext(img, slope_ths=0, width_ths=0)
        return picture_results
    
    def imagepreproc(self, img,
                     threshold=128,
                     sharping=ImageFilter.SHARPEN):
        try:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Increase contrast (1.0 means no change)

            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(2.0)  # Increase brightness

            img = img.convert("L")       # image binarization
        except Exception as e:
            print("Warning: catch some error {}".format(e))
    
        return img

    def readdisclaimer(self, img, extracted_text):
        detection_results = []

        dislcaimer_results = self.OCR_Reader.readtext(img,paragraph=True)
        box_coordinates,max_paragraph = [],''
        for box,paragraph in dislcaimer_results:
            if len(paragraph) > len(max_paragraph):
                max_paragraph = paragraph
                box_coordinates = box

        scaling_factor= 1.5
        crop_box      = *box_coordinates[0],*box_coordinates[2]
        cropped_image = img.crop(crop_box)
        resized_image = cropped_image.resize((int(cropped_image.width*scaling_factor), int(cropped_image.height)),
                                            Image.BILINEAR)
        
        max_paragraph = self.picture_read.readtext(resized_image,paragraph=True)
        
        for box,text,_ in extracted_text:
            if not boxes_overlap(bounding_box(box), bounding_box(box_coordinates)):
                detection_results.append((box,text))
        
        return detection_results

    def disclaimerpreproc(self, img, crop_box,
                          upscale_method=Image.BILINEAR,
                          scaling_factor=1.5,
                          denoising=ImageFilter.MedianFilter(size=3),
                          sharping=ImageFilter.SHARPEN):
        """
            Enhance disclaimers text detection quality
        """
        scaling_factor = 1.5
        crop_box = *crop_box[0],*crop_box[2]
        cropped_image = img.crop(crop_box)
        resized_image = cropped_image.resize((int(cropped_image.width*scaling_factor), int(cropped_image.height)),
                                                Image.BILINEAR)
        
        img = resized_image

        # # binarization
        # img = img.convert("L")

        # # Increase contrast (1.0 means no change)
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.2)

        # # Increase brightness
        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(1.2)

        # # Denoising
        # img = img.filter(sharping)
        # img = img.filter(denoising)

        return np.array(img)
    
    def getstrtext(self, picture_results):
        """
            Return text as a string
        """
        picture_results_str = ''
        for item in picture_results:
            picture_results_str += item[1] + '\n'

        result = picture_results_str
        return result
    
    def readbanner(self, image):
        detection_results = []

        # Text detection
        picture_results = self.OCR_Reader.readtext(np.array(image), slope_ths=0, width_ths=0)
        
        # Disclaimer detection
        dislcaimer_results = self.OCR_Reader.readtext(np.array(image), paragraph=True)
        
        if len(dislcaimer_results) == 0:
            return picture_results, ''
        
        # Disclaimer position
        box_coordinates,max_paragraph = [],''
        for (box,paragraph) in dislcaimer_results:
            if len(paragraph) > len(max_paragraph):
                box_coordinates = box
                max_paragraph = paragraph

        # get content of a banner
        for box,text,_ in picture_results:
            if not boxes_overlap(bounding_box(box), bounding_box(box_coordinates)):
                detection_results.append((box,text))
        
        # Preproccess the image
        optimized_image = self.disclaimerpreproc(image, box_coordinates)

        # get disclaimer
        dislcaimer_results = self.OCR_Reader.readtext(optimized_image,paragraph=True)
        for (box,paragraph) in dislcaimer_results:
            if len(paragraph) > len(max_paragraph):
                box_coordinates = box
                max_paragraph = paragraph

        OCR_text = []
        for item in detection_results:
            OCR_text.append(item[1])

        return OCR_text, max_paragraph
