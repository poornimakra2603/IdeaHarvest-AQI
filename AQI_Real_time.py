import requests
import json
import random
import sqlite3
city = 'Pallavaram'
stationcode = random.randint(1,100)
pollutantid = random.randint(1,10)
covalue = ''
o3value = ''
api_url = 'https://api.api-ninjas.com/v1/airquality?city={}'.format(city)
response = requests.get(api_url, headers={'X-Api-Key': 'VIJwWwsCIHEZOcUVRLb1/Q==dbRSG8sUuxC3ibB5'})
if response.status_code == requests.codes.ok:
    print(response.text)
    data = json.loads(response.text)
    covalue = data['CO']['concentration']
    o3value = data['O3']['concentration']
    conn = sqlite3.connect('database_aqi.db') 
    c = conn.cursor()
    with sqlite3.connect("database_aqi.db") as users: 
            cursor = users.cursor()
            cursor.execute("SELECT * FROM cities_data where city_name = ?",(city,))
            data_check = cursor.fetchall()
            if(len(data_check)> 0):
                cursor.execute("UPDATE cities_data SET pollutant_max = ? , pollutant_min = ? WHERE city_name = ?",(covalue,o3value,city)) 
            else:
                cursor.execute("INSERT INTO cities_data (city_name,pollutant_id,pollutant_max,pollutant_min,station_code) VALUES (?,?,?,?,?)", (city,pollutantid,covalue,o3value,stationcode)) 
            users.commit()

        
else:
    print("Error:", response.status_code, response.text)