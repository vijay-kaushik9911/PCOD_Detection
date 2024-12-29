from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing import image  # type: ignore
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore
from io import BytesIO  # Import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (assuming you've saved it as an H5 file)
ann = load_model("ml_model/ann_model.keras")
cnn = load_model("ml_model/cnn_model.keras")

# Load the scaler (saved in 'scaler.pkl')
scaler = pickle.load(open('ml_model/ann_scaler.pkl', 'rb'))

# Function for preprocessing the input image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(64, 64))  # Ensure this matches your CNN input size
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)
    return img

# Route to render the index page (Frontend form)
@app.route('/')
def index():
    return render_template('index.html')  # Points to index.html template

# Route to predict page
@app.route('/predict')
def predict_page():
    return render_template('predict.html')

# API route to handle prediction
@app.route('/predictres', methods=['POST'])
def predictres():
    try:
        # Get form data
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        irregular_periods = request.form['irregularPeriods'] == 'yes'
        acne = request.form['acne'] == 'yes'
        pregnant = request.form['pregnant'] == 'yes'
        hirsutism = request.form['hirsutism'] == 'yes'
        hair_loss = request.form['hairLoss'] == 'yes'
        fsh = float(request.form['fsh'])
        lh = float(request.form['lh'])
        tsh = float(request.form['tsh'])
        vitamin_d3 = float(request.form['vitaminD3'])
        ultrasound = request.form['ultrasound'] == 'yes'

        # Prepare input data for prediction
        input_data = np.array([[age, weight, irregular_periods, acne, pregnant, hirsutism, hair_loss, fsh, lh, tsh, vitamin_d3]])

        # Scale the input data using the previously saved scaler
        input_data_scaled = scaler.transform(input_data)

        # Check if an image is uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            # Handle image prediction (use both ANN and CNN)
            file = request.files['file']
            
            # Read the image file directly into memory
            img = preprocess_image(BytesIO(file.read()))

            # Predict using CNN
            cnn_result = cnn.predict(img)
            
            # Use ANN for input data prediction
            ann_prediction = ann.predict(input_data_scaled)

            # Combine predictions (you can adjust this logic)
            final_prediction = (ann_prediction*0.6 + cnn_result*0.4)

            print("\n\n\n **********************************\n\n\n")
            print("Prediction % :  ",final_prediction[0][0])
            print("\n\n\n **********************************\n\n\n")

            # Check if prediction is above threshold (example threshold = 0.5)
            prediction = 'not_infected' if final_prediction[0][0] > 0.5 else 'infected'

        else:
            # Use only ANN for prediction (if no image is uploaded)
            ann_prediction = ann.predict(input_data_scaled)
            
            # Check prediction result for ANN
            prediction = 'not_infected' if ann_prediction[0][0] > 0.5 else 'infected'

        # Render the results page with the prediction result
        return render_template('results.html', prediction=prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)