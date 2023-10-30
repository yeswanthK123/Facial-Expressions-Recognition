import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image 
from flask import Flask, render_template, Response, request, redirect, flash
from tensorflow.keras.models import model_from_json 

# Load the model
model = model_from_json(open("jsn_model.json", "r").read())  
model.load_weights('weights_model1.h5')  

# Load the Haar Cascade Classifier for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

# Configure the app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the routes
@app.route('/')
def start():
    return render_template('index.html')

# Define the function to generate frames
def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, test_img = camera.read()
        if not success:
            break
        else:
            # Your video frame processing code goes here
            # ...

            ret, buffer = cv2.imencode('.jpg', test_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the function for real-time analysis
def emotion_analysis(img):
    # Your emotion analysis code goes here
    # ...

@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = emotion_analysis(filename)

            if len(result) == 1:
                return render_template('no_prediction.html', orig=result[0])

            return render_template('prediction.html', orig=result[0], pred=result[1])

if __name__ == '__main__':
    app.run(debug=True)
