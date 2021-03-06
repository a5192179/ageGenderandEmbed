from flask import Flask
from flask import request
import json
from waitress import serve
import base64
import cv2
import numpy as np
import sys
sys.path.append('.')
from web import processJson
# import web.processJson
from algoModule.estimateAgeGender.algo import estimateAgeGender
from algoModule.embedFace.algo import embedFace
import time

# =============================
# txtPath = './code/dependence.txt'
# with open(newFilePath, "w") as f:
#                 f.write(newTest)
# =============================
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    return '<h1>Hello</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

@app.route('/aidata', methods=['POST'])
def print_info():
    data = json.loads(request.get_data(as_text=True)) # 获取前端POST请求传过来的json数据
    print(data)
    return '<h1>ai data</h1>'

@app.route('/image-encode', methods=['GET', 'POST'])
def my_encode():
    image = 'data,base64.....'
    myEncode(image)
    return 'encode image'

@app.route('/algo/v1/queryPersonAttribute', methods=['POST'])
def queryPersonAttribute():
    data = json.loads(request.get_data())
    print(type(data))
    ts = time.time()
    with ageGenderEstimater.graph.as_default():
        age, gender = processJson.estimateAgeGender(data, ageGenderEstimater)
    print('time:', time.time() - ts)
    print("age:", age, "gender:", gender)
    results = {}
    results['age'] = age
    results['gender'] = gender
    
    return json.dumps(results)

@app.route('/algo/v1/queryImageFeature', methods=['POST'])
def queryImageFeature():
    data = json.loads(request.get_data())
    print(type(data))
    with faceEmbedder.graph.as_default():
        imgName, feature = processJson.embedFace(data, faceEmbedder)
    results = {}
    results[imgName] = feature.tolist()
    return json.dumps(results)

@app.route('/algo/v1/compareSimilarity', methods=['POST'])
def compareSimilarity ():
    data = json.loads(request.get_data())
    print(type(data))
    dist = processJson.compareSimilarity(data)
    results = {}
    results['distance'] = dist
    print(dist)
    
    return json.dumps(results)

def initModel():
    global ageGenderEstimater
    ageGenderEstimater = estimateAgeGender.ageGenderEstimater()
    global faceEmbedder
    faceEmbedder = embedFace.faceEmbedder()

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=9000, debug=True)
    initModel()
    serve(app, host='0.0.0.0', port=9000)