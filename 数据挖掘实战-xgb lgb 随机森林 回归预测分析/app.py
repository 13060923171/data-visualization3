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
    # 加载xgb模型
    model = joblib.load('model_XGB.pkl')
    if request.method == 'POST':
        if request.form.get('brand') != None and request.form.get('mileage') != None and request.form.get('company') != None and request.form.get('price') != None \
                and request.form.get('form') != None and request.form.get('genre') != None and request.form.get('body') != None and request.form.get('warranty') != None \
                and request.form.get('power') != None and request.form.get('transfer') != None and request.form.get('production') != None:
            brand = request.form.get('brand')
            brand = int(brand)
            mileage = request.form.get('mileage')
            mileage = float(mileage)
            company = request.form.get('company')
            company = int(company)
            price = request.form.get('price')
            price = float(price)
            form = request.form.get('form')
            form = int(form)
            genre = request.form.get('genre')
            genre = int(genre)
            body = request.form.get('body')
            body = int(body)
            warranty = request.form.get('warranty')
            warranty = int(warranty)
            power = request.form.get('power')
            power = int(power)
            transfer = request.form.get('transfer')
            transfer = int(transfer)
            production = request.form.get('production')
            production = int(production)
        data = [[brand,mileage,company,production,price,form,genre,body,warranty,power,transfer]]
        y_pre = model.predict(data)
        predict_price = y_pre[0]
        predict_price = round(predict_price,2)
        return render_template('predict.html',brand=brand,mileage=mileage,
                               company=company,price=price,form=form,genre=genre,
                               body=body,warranty=warranty,power=power,transfer=transfer,production=production,predict_price=predict_price)

    return render_template('pricing_forecast.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5050',debug=True)


