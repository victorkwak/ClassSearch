import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from flask import Flask, request, Response
from flask import render_template
import numpy as np
from fastText import load_model
import matplotlib.pyplot as plt
import json
import cPickle
import io
import base64


class Classifier(object):
    def __init__(self):
	self.encoder = cPickle.loads(open("encoder2.7.p", "rb").read())
        self.classifier = load_model('models/fasttext.bin')

    def predict(self, title):
        labels, values = self.classifier.predict(title, 10)
        labels = list(map(lambda x: int(x.strip('__label__')), labels))
        labels = self.encoder.inverse_transform(labels).tolist()
        return dict(zip(*(labels, values)))


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
    #  response = make_response(output.getvalue())
    #  response.mimetype = 'image/png'
    plot_url = base64.b64encode(output.getvalue()).decode()
    #  https://stackoverflow.com/questions/20836766/how-do-i-remove-broken-image-box
    return render_template('classify.html', chart=plot_url)


@app.route('/classify_this_post_api', methods=['POST'])
def classify_this_post_api():
    try:
        post_title = request.form['post_title']
    except KeyError:
        return ("BAD-FORM, 'post_title' NOT FOUND", 400)
    data = classifier.predict(str(post_title))
    js = json.dumps(data)
    return Response(js, status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
