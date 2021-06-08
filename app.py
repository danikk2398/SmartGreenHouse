import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
MODEL_PATH = './models/model_inception.h5'

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

#print('Model loaded. Check http://127.0.0.1:5000/')
# Load your trained model
#model = load_model(MODEL_PATH)


# Model saved with Keras model.save()

# Load your own trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_get):
	#img_get = img_get.resize((224, 224))
	#img_get.save("./uploads/image.png")
	#img_ref="./uploads/image.png"
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

 #    # Be careful how your trained model deals with the input
 #    # otherwise, it won't make correct prediction!
 #    x = preprocess_input(x, mode='tf')

 #    preds = model.predict(x)
 #    return preds
	#print(img)
	img = image.load_img(img_get, target_size=(224, 224))
	x = image.img_to_array(img)
	
	#x = np.true_divide(x, 255)
	## Scaling
	x=x/255
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#model=load_model(MODEL_PATH)
	preds = model.predict(x)
	#model=''
	print(preds)
	preds=np.argmax(preds, axis=1)
	if preds==0:
		preds="Bacterial_spot"
	elif preds==1:
		preds="Early_blight"
	elif preds==2:
		preds="Late_blight"
	elif preds==3:
		preds="Leaf_Mold"
	elif preds==4:
		preds="Septoria_leaf_spot"
	elif preds==5:
		preds="Spider_mites Two-spotted_spider_mite"
	elif preds==6:
		preds="Target_Spot"
	elif preds==7:
		preds="Tomato_Yellow_Leaf_Curl_Virus"
	elif preds==8:
		preds="Tomato_mosaic_virus"
	else:
		preds="healthy"
	print(preds)
	return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        #print(request.json)
        img = base64_to_pil(request.json)
        #print("IMG!!!")
        #print(img)
        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        
        preds = model_predict("./uploads/image.png")
        os.remove("./uploads/image.png")
        # Process your result for human
        #pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        #result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=preds, probability=preds)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
