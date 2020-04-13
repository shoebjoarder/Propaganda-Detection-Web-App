from flask import Flask, render_template, url_for, request
import pickle as pk

PROPA_DATADIR = './data'
PROPA_MODEL = ['best_model.sav', 'vectorizer.sav']
app = Flask(__name__)


class PropagandaNews(object):

    def __init__(self):
        self.datadir = PROPA_DATADIR
        self.model_objects = PROPA_MODEL

    def predict_propaganda(self, news_text):
        best_model = pk.load(open(self.datadir + '/' + self.model_objects[0], 'rb'))
        best_vector = pk.load(open(self.datadir + '/' + self.model_objects[1], 'rb'))

        article_tf = best_vector.transform([news_text])
        prop_predict = best_model.predict(article_tf)

        if prop_predict == ['non-propaganda']:
            return 0
        elif prop_predict == ['propaganda']:
            return 1


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        news_propaganda = PropagandaNews()
        my_prediction = news_propaganda.predict_propaganda(message)
    return render_template('prediction.html', prediction=my_prediction)


if __name__ == '__main__':
    #    app.run(host='0.0.0.0', port=4000)
    app.run(port=4000)
