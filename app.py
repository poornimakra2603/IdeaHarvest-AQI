from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import random
import pandas as pd
import sqlite3

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

connect = sqlite3.connect('database_aqi.db')

@app.route('/')
def hello_world():
    return render_template("Air_Quality_Prediction.html")

@app.route('/home')
def home():
    return render_template("Air_Quality_Prediction.html")

@app.route('/updates')
def updates():
    return render_template("Updates.html")

@app.route('/savedetails',methods=['POST','GET'])
def savedetails():
    if request.method == 'POST': 
        name = request.form['name'] 
        email = request.form['Email'] 
        disease = request.form['Disease'] 
  
        with sqlite3.connect("database_aqi.db") as users: 
            cursor = users.cursor() 
            cursor.execute("INSERT INTO user_data (user_name,Email,Disease) VALUES (?,?,?)", (name, email, disease)) 
            users.commit() 
        return render_template("Success.html",Smessage = "Records Inserted Successfully")
    else: 
        return render_template('Failed.html') 
'''@app.route('/Success')
def Success():
    return render_template("Success.html")'''

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST': 
        email = request.form['Email']
    connect = sqlite3.connect('database_aqi.db') 
    cursor = connect.cursor() 
    cursor.execute("SELECT * FROM user_data where email = ?",(email,))
    emaildata = cursor.fetchall()
    if(len(emaildata) > 0):
        disease_ofuser = ''
        for row in emaildata:
            disease_ofuser = row[2]
        pollutantid = 0
        pollutantmax = 0
        pollutantmin = 0
        stationcode = 0
        cityname =''
        if request.method == 'POST': 
            cityname = request.form['Cityname']

        connect = sqlite3.connect('database_aqi.db') 
        cursor = connect.cursor() 
        cursor.execute("SELECT * FROM cities_data where city_name = ?",(cityname,))
        data = cursor.fetchall()
        if(len(data) > 0):
            for row in data:
                pollutantid = row[1]
                pollutantmax = row[2]
                pollutantmin = row[3]
                stationcode = row[4]
            normalist = [pollutantmin,pollutantmin,stationcode,pollutantid]
            features = pd.DataFrame({'pollutant_min' : [pollutantmin],'pollutant_max' : [pollutantmax], 'station_code' : [stationcode], 'pollutant_id' : [pollutantid]})
            prediction = model.predict(features)
            predoutput = prediction.astype(int)
    
            output = list(map('{:.2f}'.format,predoutput))
            output_string = ",".join(str(element) for element in output)
            res = eval(output_string)
        else:
            return render_template("Nocity.html")
    else:
        return render_template('Updates.html',notfill='Please fill this form first')



    if res>0 and res<=50:
        return render_template('Safe.html',pred="GOOD!!\nAir Quality Index is {}.This air gives only MINIMAL IMPACT".format(res))

    elif res>=51 and res<=100:
        return render_template('Satisfactory.html',pred="SATISFACTORY!!\nAir Quality Index is {}.This air gives only MINOR BREATHING DISCOMFORT TO SENSITIVE PEOPLE".format(res))

    elif res>=101 and res<=200:
        return render_template('Moderate.html',pred="MODERATE!!\nAir Quality Index is {}.This air gives only BREATHING DISCOMFORT TO THE PEOPLE WITH LUNGS,ASTHMA AND HEART DISEASES".format(res))

    elif res>=201 and res<=300:
        return render_template('Poor.html',pred="POOR!!\nAir Quality Index is {}.This air gives only BREATHING DISCOMFORT TO MOST PEOPLE ON PROLONGED MEASURE".format(res))

    elif res>=301 and res<=400:
        return render_template('Very Poor.html',pred="VERY POOR!!\nAir Quality Index is {}.This air gives only RESPIRATORY ILLNESS ON PROLONGED EXPOSURE".format(res))

    elif res>=401 and res<=500:
        return render_template('NotSafe.html',pred="SEVERE!!\nAir Quality Index is {}.This air affects HEALTHY PEOPLE AND SERIOUSLY IMPACTS THOSE WITH EXISTING DISEASES".format(res))
    
if __name__ == '__main__':
    app.run(debug=True)
