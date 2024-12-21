from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pdfplumber
from extraction import (
    convert_pdf_to_images, 
    load_yolo_model, 
    initialize_ocr, 
    is_scanned_pdf, 
    process_images,
    get_pdf_page_count,
)

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            is_scanned = is_scanned_pdf(file_path)
            page_count = get_pdf_page_count(file_path)

            pdf_type = "SA104F Full Response" if page_count > 2 else "SA104 Short Response"

            images = convert_pdf_to_images(file_path, is_scanned)
            model_path = "best (7).pt"
            model = load_yolo_model(model_path)
            
            ocr_full = initialize_ocr(is_simple=False)
            ocr_simple = initialize_ocr(is_simple=True)

            detected_data = process_images(images, model, ocr_full, ocr_simple, file_path, is_scanned, pdf_type)

            return jsonify({
                'pdf_type': pdf_type,
                'detected_data': detected_data
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)