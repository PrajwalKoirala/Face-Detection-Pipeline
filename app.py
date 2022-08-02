import base64
import io
import os

import numpy as np
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
from detector import Comparer
comparer = Comparer()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def dogorcat(img):

    return str(my_response)



@app.route('/ajax', methods=['POST'])
def ajax():
    a = request.form['img']
    r =  base64.b64decode(a.split(",")[1])
    image = Image.open(io.BytesIO(r))
    image = np.array(image)
    classes = comparer.find_match(image)
    my_response = {
        "prediction": {
            "label": str(classes[0])
        }
    }
    print(classes)
    return jsonify(my_response)




app.run(host=os.getenv('IP', '0.0.0.0'),
        port=int(os.getenv('PORT', 4444)))
