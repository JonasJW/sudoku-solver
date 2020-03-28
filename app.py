from flask import Flask, render_template, request, jsonify
from solver import solve
from sudokuDetector import solveSud
import json
import numpy
import cv2
from urllib.request import urlopen

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=["GET", "POST"])
def result():
    if request.method == "POST":
        if request.files:
            filestr = image = request.files["image"].read()
            npimg = numpy.fromstring(filestr, numpy.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            return solveAndSend(img)

    return 'ERROR'

@app.route('/link', methods=["GET", "POST"])
def solveLink():
    if request.method == "POST":
        link = request.form['link']
        req = urlopen(link)
        arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
        img = cv2.imdecode(arr, -1)

        return solveAndSend(img)
        
    
    return "ERROR"

def solveAndSend(img):
    res, img_b64, detected = solveSud(img)
    # print(res)
    if res is not False:
    #     res = splitResultInRows(res)
        res = list(res.values())
        res = list(map(int, res))

    return render_template('result.html', result=json.dumps(res), img=img_b64, detected=json.dumps(detected.tolist()))


def splitResultInRows(res):
    res = list(res.values())
    return [res[i:i+9] for i in range(0, len(res), 9)]


if __name__ == '__main__':
    app.debug = True
    app.run()