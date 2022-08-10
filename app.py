import base64
import os
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from detector import Comparer

comparer = Comparer()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ajax', methods=['POST'])
def ajax():
    a = request.form['img']
    r = base64.b64decode(a.split(",")[1])
    image = Image.open(io.BytesIO(r))
    image = np.array(image)
    plot_url, faces, classes = comparer.find_matches(image)
    my_response = {
        "prediction": str(classes)
    }
    print(my_response)
    return jsonify({"image": 'data:image/png;base64,' + plot_url})


app.run()
