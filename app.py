from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils.predictor import predict_waste_type
from utils.bin_manager import get_bin_name, get_bin_status
from utils.notify import send_bin_full_alert
import shutil
import uuid
import base64
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Or custom-trained model

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-123')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

BIN_FOLDERS = {
    'Red': os.path.join(UPLOAD_FOLDER, 'Red'),
    'Green': os.path.join(UPLOAD_FOLDER, 'Green'),
    'Blue': os.path.join(UPLOAD_FOLDER, 'Blue')
}

for folder in BIN_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    bin_status = get_bin_status(BIN_FOLDERS)
    return render_template('index.html', bin_status=bin_status)

@app.route('/upload/<bin_name>', methods=['GET', 'POST'])
def upload(bin_name):
    bin_name = bin_name.capitalize()
    if bin_name not in BIN_FOLDERS:
        return "Bin not found", 404

    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        files = request.files.getlist('files') if 'files' in request.files else [request.files['file']]
        
        if not files or all(file.filename == '' for file in files):
            if is_ajax:
                return jsonify({'error': 'No files selected'}), 400
            flash('No files selected')
            return redirect(request.url)

        results = []
        for file in files:
            result = {
                'filename': file.filename,
                'status': 'pending'
            }
            
            try:
                if file.filename == '':
                    result.update({'status': 'no_file', 'message': 'No file selected'})
                    results.append(result)
                    continue

                if not allowed_file(file.filename):
                    result.update({'status': 'invalid_type', 'message': 'Invalid file type'})
                    results.append(result)
                    continue

                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > MAX_FILE_SIZE:
                    result.update({'status': 'too_large', 'message': 'File too large'})
                    results.append(result)
                    continue

                filename = secure_filename(file.filename)
                unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
                temp_path = os.path.join(UPLOAD_FOLDER, unique_name)
                file.save(temp_path)

                category, confidence = predict_waste_type(temp_path)
                
                if not category:
                    os.remove(temp_path)
                    result.update({'status': 'classification_failed', 'message': 'Could not classify waste type'})
                    results.append(result)
                    continue

                predicted_bin = get_bin_name(category)
                
                if predicted_bin == bin_name:
                    final_path = os.path.join(BIN_FOLDERS[bin_name], unique_name)
                    os.rename(temp_path, final_path)
                    result.update({
                        'status': 'success',
                        'category': category,
                        'bin': predicted_bin,
                        'confidence': confidence
                    })

                    # ✅ Check if bin is full after upload and send SMS+Email
                    bin_status = get_bin_status(BIN_FOLDERS)
                    if bin_status[bin_name]['full']:
                        send_bin_full_alert(bin_name)

                else:
                    os.remove(temp_path)
                    result.update({
                        'status': 'wrong_bin',
                        'category': category,
                        'correct_bin': predicted_bin,
                        'confidence': confidence
                    })
                
                results.append(result)

            except Exception as e:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
                result.update({'status': 'error', 'message': str(e)})
                results.append(result)

        if is_ajax:
            return jsonify({'results': results})
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        wrong_bin_count = sum(1 for r in results if r.get('status') == 'wrong_bin')
        error_count = len(results) - success_count - wrong_bin_count

        if success_count > 0:
            flash(f"{success_count} file(s) uploaded successfully", 'success')
        if wrong_bin_count > 0:
            flash(f"{wrong_bin_count} file(s) belong to different bins", 'warning')
        if error_count > 0:
            flash(f"{error_count} file(s) had errors", 'error')
        
        return redirect(url_for('view_bin', bin_name=bin_name))

    return render_template('upload.html', bin_name=bin_name)

@app.route('/bin/<bin_name>')
def view_bin(bin_name):
    bin_name = bin_name.capitalize()
    if bin_name not in BIN_FOLDERS:
        return "Bin not found", 404

    try:
        images = [f for f in os.listdir(BIN_FOLDERS[bin_name]) 
                 if os.path.isfile(os.path.join(BIN_FOLDERS[bin_name], f)) and
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        images.sort(key=lambda x: os.path.getmtime(os.path.join(BIN_FOLDERS[bin_name], x)), reverse=True)
    except Exception as e:
        flash(f"Error reading bin contents: {str(e)}", 'error')
        images = []

    return render_template('bin.html',
                         bin_name=bin_name,
                         images=images,
                         image_count=len(images))

@app.route('/manual_upload')
def manual_upload():
    return render_template('manual_upload.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

        temp_filename = f"webcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        cv2.imwrite(temp_path, img)

        category, confidence = predict_waste_type(temp_path)
        os.remove(temp_path)

        if not category:
            return jsonify({'status': 'error', 'message': 'Classification failed'}), 400

        predicted_bin = get_bin_name(category)
        if not predicted_bin:
            return jsonify({'status': 'error', 'message': 'Bin mapping failed'}), 400

        return jsonify({
            'status': 'success',
            'category': category,
            'bin': predicted_bin,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
