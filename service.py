from flask import Flask, request, Response
from flask import render_template, make_response
import pandas as pd
import numpy as np
import json, pickle
import fastText as fasttext
from fastText import load_model
import matplotlib.pyplot as plt; plt.rcdefaults()
import io, base64

class Classifier(object):
    def __init__(self):
        self.classifier = load_model('models/fasttext.bin')
        self.encoder = pickle.load(open( "encoder.p","rb"))

    def predict(self, title):
        labels, values = self.classifier.predict(title, 10)
        labels = list(map(lambda x: int(x.strip('__label__')), labels))
        labels = self.encoder.inverse_transform(labels).tolist()
        return dict(zip(*(labels,values)))

app = Flask(__name__)
classifier = Classifier()

@app.route('/')
def index():
    return render_template('classify.html')

@app.route('/plot_chart', methods=['GET'])
def plot_chart():
    post_title = request.args['post_title']
    data = classifier.predict(str(post_title))
    labels = list(data.keys())
    values = list(data.values())
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values, color='#347cef')
    plt.yticks(y_pos, labels)
    plt.xlabel('Probability')
    plt.title('Sub-reddits')
    plt.tight_layout()
    output = io.BytesIO()
    plt.savefig(output)
    #response = make_response(output.getvalue())
    #response.mimetype = 'image/png'
    plot_url = base64.b64encode(output.getvalue()).decode()
    #https://stackoverflow.com/questions/20836766/how-do-i-remove-broken-image-box
    return render_template('classify.html', chart=plot_url)

@app.route('/classify_this_post_api',methods=['POST'])
def classify_this_post_api():
    try:
        post_title = request.form['post_title']
    except KeyError:
        return ("BAD-FORM, 'post_title' NOT FOUND", 400)
    data = classifier.predict(str(post_title))
    js = json.dumps(data)
    return Response(js, status=200, mimetype='application/json')

if __name__ == '__main__':
    #don't use debug since it would create 2 schedulers due to reload functionality
    #app.run(host="0.0.0.0", port=8019)
    app.run(host='localhost',port=8080, debug=True)
