from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from numpy import vectorize
from wtforms import StringField, SubmitField, FileField
from wtforms.validators import DataRequired
import pickle

from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from ensembles import GradientBoostingMSE
import numpy as np
from flask_bootstrap import Bootstrap

import re
from nltk.stem.snowball import RussianStemmer

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'secret_key'
Bootstrap(app)

file = open('model_small.pkl', 'rb')
data = pickle.load(file)

vectorizer, list_of_topics, list_of_models = data

vectorizer: CountVectorizer
list_of_topics: List[str]
list_of_models: List[GradientBoostingMSE]

class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    submit = SubmitField('Get Result')

def guess(text: str):
    stem_rus = RussianStemmer(False)

    out = ''
    for word in re.sub('[^{0-9а-яА-Я}]', ' ', text).lower().split(' '):
        out += stem_rus.stem(word) + ' '
    print('text proceeded')

    X_from_user = vectorizer.transform([out]).toarray()
    list_of_predictions: List[float] = []

    for gb in list_of_models:
        list_of_predictions.append(gb.predict(X_from_user))
    print('predictions made')

    rating = sorted(list(zip(list_of_predictions, list_of_topics)))[::-1]
    print('rating created')
    _, top_1 = rating[0]
    _, top_2 = rating[1]
    _, top_3 = rating[2]
    return top_1, top_2, top_3

@app.route('/', methods=['GET', 'POST'])
def get_text_score():
    text_form = TextForm()
    if text_form.validate_on_submit():
        text_data = text_form.text.data
        text_form.text.data = ''
        top_1, top_2, top_3 = guess(text_data)
        return render_template('answer.html', idea_1=top_1, idea_2=top_2, idea_3=top_3)
    return render_template('input.html', text_data=text_form)


app.run(host='0.0.0.0', port=5000, debug=True)