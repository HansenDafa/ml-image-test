from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import os
from model import cifar10_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img_array = load_and_preprocess_image(file_path)
            predicted_class = cifar10_model.predict(img_array)
            return render_template('index.html', filename=file.filename, predicted_class=predicted_class)
    return render_template('index.html', filename=None, predicted_class=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
