from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLOv8 model
model = YOLO('best.pt')

def allowed_file(filename):
    """Check if uploaded file is valid."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint for detecting fruits in an image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform detection
        results = model.predict(source=filepath, save=True)
        
        # Get rendered image from results
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        result_image = results[0].plot()
        
        # Save the result image
        Image.fromarray(np.uint8(result_image)).save(result_image_path)
        
        # Return the result image filename
        return jsonify({"result_image": f"uploads/result_{filename}"}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded result image."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
