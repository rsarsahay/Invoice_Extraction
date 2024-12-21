import os
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO
from paddleocr import PaddleOCR
import pdfplumber
import re
import io

def convert_pdf_to_images(pdf_path, is_scanned):
    if is_scanned:
        return convert_from_path(pdf_path, dpi=700, size=(5785, 8177))
    else:
        return convert_from_path(pdf_path, dpi=200, size=(595.276, 851.89))

def load_yolo_model(model_path):
    return YOLO(model_path)

def initialize_ocr(is_simple=False):
    if is_simple:
        return PaddleOCR(
            use_angle_cls=False,
            lang='en'
        )
    return PaddleOCR(
        use_angle_cls=False,
        lang='en',
        e2e_algorithm='PGNet',
        det_sast_score_thresh=0.8,
        e2e_pgnet_mode='fast',
        det_sast_nms_thresh=0.2,
        max_text_length=200,
        det_limit_side_len=4000,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=3.0,
        det_limit_type='max',
        cls_image_shape='3, 480, 1080',
        drop_score=0.3,
        rec_image_inverse=False,
        rec_image_shape='3, 4160, 6000',
        rec_algorithm='CRNN',
        use_space_char=True,
        show_log=False,
        enable_mkldnn=True,
    )

def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def process_text(text, class_name, is_digital_pdf=False):
    if text is None:
        return None

    preserve_fields = ['Box2', 'Name', 'FormType', 'Year']

    if class_name == 'Year':
        cleaned_year_text = text.replace("z", "").strip()
        cleaned_year_text = re.sub(r'\s+', ' ', cleaned_year_text) 
        return cleaned_year_text

    if 'X' in text:
        return 'X'

    if class_name in preserve_fields:
        return text.strip()

    if class_name in ('Box3', 'Box4', 'Box6', 'Box7'):
        date_pattern = r'(\d{1,2})(\d{1,2})(\d{4})'
        match = re.search(date_pattern, text.replace(" ", ""))
        if match:
            day = match.group(1).zfill(2)
            month = match.group(2).zfill(2)
            year = match.group(3)
            return f"{day} {month} {year}"
        return text.strip()

    elif class_name == 'Box32':
        return ' '.join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', text))

    cleaned_text = ''.join(char for char in text if char.isdigit() or char in ['.', 'X', 'O'])

    if is_digital_pdf and cleaned_text.endswith('00'):
        cleaned_text = cleaned_text[:-2] + ".00"
    elif cleaned_text.endswith('00'): 
        cleaned_text = cleaned_text[:-2] + ".00"

    return cleaned_text if cleaned_text else text

def extract_text_with_pdfplumber(pdf_path, page_number, coords):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        cropped_page = page.crop(coords)
        text = cropped_page.extract_text()
        return text.strip() if text else None 

def extract_text_with_ocr(image, ocr, coordinates, class_name):
    print(image.size)
    cropped_img = image.crop(coordinates)
    img_array = np.array(cropped_img)
    ocr_results = ocr.ocr(img_array, cls=True)

    extracted_texts = []

    if ocr_results and isinstance(ocr_results, list):
        if class_name == 'Box32':
            for line in ocr_results:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        extracted_texts.append(text)
        else:
            word_positions = []
            for line in ocr_results:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        bbox = word_info[0]
                        x_position = bbox[0][0]  
                        word_positions.append((x_position, text))

            word_positions.sort(key=lambda x: x[0])

            extracted_texts = [word for _, word in word_positions]

    return " ".join(extracted_texts) if extracted_texts else None

def process_images(images, model, ocr_full, ocr_simple, pdf_path, is_scanned, pdf_type):
    detected_data = {}

    for i, image in enumerate(images):
        results = list(model(image, stream=True)) 

        if len(results) == 0:
            continue 

        detections = []

        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls

            for bbox, confidence, class_id in zip(boxes, confidences, classes):
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence.item(),
                    'class_id': int(class_id),
                    'class_name': result.names[int(class_id)]
                })

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            x0, y0, x1, y1 = map(int, map(round, [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]))
            
            if is_scanned:
                use_simple_ocr = pdf_type == "SA104 Short Response" and class_name == "Box32"
                current_ocr = ocr_simple if use_simple_ocr else ocr_full
                extracted_text = extract_text_with_ocr(image, current_ocr, (x0, y0, x1, y1), class_name)
            else:
                extracted_text = extract_text_with_pdfplumber(pdf_path, i, (x0, y0, x1, y1))

            if extracted_text is None:
                extracted_text = ""  

            processed_text = process_text(extracted_text, class_name, is_digital_pdf=not is_scanned)
            detected_data = replace_zero_values(detected_data)

            if "are not in use" not in extracted_text and confidence >= 0.40:
                detected_data[class_name] = processed_text 

    return detected_data

def replace_zero_values(data):
    for key, value in data.items():
        if value == ".00" or value == "0":
            data[key] = None
    return data

def is_scanned_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    page = pdf.pages[0]
    text = page.extract_text()
    if len(text) == 0:
        return True
    return False

def get_pdf_page_count(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)
