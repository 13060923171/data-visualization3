#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template,request
import joblib
from sklearn import preprocessing
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    global name,login
    # 加载LabelEncoder模型
    loaded_le = joblib.load('label_encoder_model.pkl')
    # 加载TfidfVectorizer模型
    loaded_vectorizer = joblib.load('tfidf_vectorizer_model.pkl')
    loaded_model = joblib.load('best_classifier_model.pkl')
    if request.method == 'POST':
        if request.form.get('comment') != None:
            comment = request.form.get('comment')
            content_vectorized = loaded_vectorizer.transform([comment])
            personality_pred = loaded_model.predict(content_vectorized)
            predicted_personality = loaded_le.inverse_transform(personality_pred)
            return render_template('personality.html',comment=comment,personality=predicted_personality[0])

    return render_template('personality_predict.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5050',debug=True)


