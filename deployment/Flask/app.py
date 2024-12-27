from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments

import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model("3D_UNet_weights_models_final4_-00026-0.18519.keras", compile=False)

def predict_and_visualize(image_path):
    test_img = np.load(image_path)
    test_img_input = np.expand_dims(test_img, axis=0)
    test_img_prediction = model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_img_prediction, axis=4)[0,:,:,:]

    # Find the slice with the highest tumor probability
    tumor_slices = np.sum((test_prediction_argmax == 1) | (test_prediction_argmax == 2) | (test_prediction_argmax == 3), axis=(0, 1))
    slice_with_max_tumor = np.argmax(tumor_slices)

    # Visualize the slice with the highest tumor probability
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.title(f'Testing Image - Slice {slice_with_max_tumor}')
    plt.imshow(test_img[:,:, slice_with_max_tumor, 1], cmap='grey')

    plt.subplot(1, 2, 2)
    plt.title(f'Prediction - Slice {slice_with_max_tumor}')
    plt.imshow(test_prediction_argmax[:,:, slice_with_max_tumor])

    plt.tight_layout()
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    plt.savefig(output_path)
    plt.close()
    return output_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"File received: {file.filename}")
            print(f"Saving file to: {file_path}")
            file.save(file_path)

            try:
                result_image_path = predict_and_visualize(file_path)
                print(f"Result image saved at: {result_image_path}")

                # Generate URL for the static file
                result_image_url = url_for('static', filename='uploads/result.png')
                print(f"Redirecting to results.html with image URL: {result_image_url}")

                return render_template('results.html', result_image=result_image_url)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return "An error occurred during prediction. Please try again."

    print("Rendering upload.html for GET request")
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)