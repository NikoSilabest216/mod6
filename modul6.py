import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# Define the allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create a Flask web application
app = Flask(__name__)

# Set the upload folder for images and specify the maximum content length for file uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Disable caching for the Flask application
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
# Set the allowed extensions in uploaded file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the route for the home page
@app.route("/")
def index():
    return render_template('/select.html')

# Define the route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files["file"]
    # Define the chosen model and corresponding file paths
    chosen_model = 'hyperModel'
    model_dict = {'hyperModel': 'static/model/vgg.h5',
                  'LRSModel': 'static/model/vgg.h5'}
    #Checking uploaded file is supported format or nah
    if file and allowed_file(file.filename):
    # Load the chosen model
        if chosen_model in model_dict:
            model = load_model(model_dict[chosen_model])
        else:
            model = load_model(model_dict[0])


    # Save the uploaded file temporarily
        file.save(os.path.join('static', 'temp.jpg'))

    # Read and preprocess the image for prediction
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3]
                    if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

    # Perform the image prediction
        start = time.time()
        pred = model.predict(img)[0]
        labels = (pred > 0.5).astype(np.int)
        runtimes = round(time.time() - start, 4)

    # Convert prediction probabilities to percentage values
        respon_model = [round(elem * 100, 2) for elem in pred]

    # Display the prediction result
        return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg')
    else:
        return render_template('/invalid.html')

# Define the function to render the prediction result
def predict_result(model, run_time, probs, img):
    # Define the class labels for the prediction
    class_list = {'paper': 0, 'rock': 1, 'scissors': 2}

    # Get the index of the predicted class
    idx_pred = probs.index(max(probs))

    # Get the class labels
    labels = list(class_list.keys())

    # Render the result template
    return render_template('/result_select.html', labels=labels,
                           probs=probs, model=model, pred=idx_pred,
                           run_time=run_time, img=img)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=2000)
