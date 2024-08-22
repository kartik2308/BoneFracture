# your_model_module.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import os

# Load your trained models
model_path_model1 = 'C:/Users/Hp/Desktop/arya/my_flask_app/mymodel.h5'
model1 = tf.keras.models.load_model(model_path_model1)

model_path_model2 = 'C:/Users/Hp/Desktop/arya/my_flask_app/mymodel2.h5'
model2 = tf.keras.models.load_model(model_path_model2)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))  # Adjust target size as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values to be between 0 and 1

        os.remove(filepath)

        return img_array

    return None

def predict(file, model_name):
    processed_image = preprocess_image(file)

    if processed_image is not None:
        if model_name == 'Model 1':
            predictions = model1.predict(processed_image)
        elif model_name == 'Model 2':
            predictions = model2.predict(processed_image)
        else:
            # Handle other models if needed
            pass

        class_prob = predictions[0][0]
        threshold = 0.5
        predicted_class = 1 if class_prob >= threshold else 0

        return {"class": predicted_class, "probability": float(class_prob)}

    return None