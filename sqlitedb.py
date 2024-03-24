import sqlite3

conn = sqlite3.connect('database_aqi.db') 
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS cities_data
          ([city_name] TEXT PRIMARY KEY, [pollutant_id] INTEGER, [pollutant_max] INTEGER,[pollutant_min] INTEGER,[station_code] INTEGER)
          ''')
          
c.execute('''
          CREATE TABLE IF NOT EXISTS user_data
          ([user_name] TEXT, [email] TEXT PRIMARY KEY,[Disease] TEXT)
          ''')
with sqlite3.connect("database_aqi.db") as users: 
            cursor = users.cursor() 
            cursor.execute("INSERT INTO cities_data (city_name,pollutant_id,pollutant_max,pollutant_min,station_code) VALUES (?,?,?,?,?)", ('Trichy',30,70,20,33)) 
            users.commit()


'''c.execute("SELECT * FROM user_data where email = ?",('krboobesh@gmail.com',))
emaildata = c.fetchall()
l = [1, 2, 3]

s = [str(integer) for integer in emaildata]
a_string = "".join(s)

res = a_string
res1 = int(''.join(map(str, res)))

print(res)
print(res1)
print(type(res))
print(type(res1))
print(emaildata)
print(type(emaildata))
conn.commit()'''

#int_features=[int(x) for x in request.form.values()]
#final=[np.array(int_features)]
#print(int_features)
#print(final)
#prediction=model.predict_proba(final)
#output='{0:.{1}f}'.format(prediction[0][1], 2)
#float_features = [float(x) for x in request.form.values()]
#float_features = np.array(float_features)
#prediction = model.predict(float_features.reshape(-1,1))
#output = '{0:.{1}f}'.format(prediction, 2)
#pollutantmin = random.randint(40,80)
#pollutantmax = random.randint(60,202)
#stationcode = random.randint(1,500)
#pollutantid = random.randint(1,10)