from os import name
from flask import Flask, redirect, url_for, render_template, request, session
from datetime import datetime
import os
import random
import matplotlib.pyplot as plt
import torch
import tensorflow.lite as tflite
import cv2
import numpy as np

finf = []
finc = []

def flat(lis):
	flatList = []
	# Iterate with outer list
	for element in lis:
		if type(element) is list:
			# Check if type is list than iterate through the sublist
			for item in element:
				flatList.append(item)
		else:
			flatList.append(element)
	return flatList

def load_random_image(folder_path):
  images = []
  for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
      images.append(os.path.join(folder_path, file_name))

  random_image = random.choice(images)
  return random_image

def detect():
    folder_path = r"C:\Users\Jegadit\Desktop\root\pah\works\python\AmritaCanteenApp\TableAndCrowd\_datasets\archive\frames"
    random_image_path = load_random_image(folder_path)

    TLmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    img = plt.imread(random_image_path)
    results = TLmodel(img)
    detections = results.pred[0][results.pred[0][:, -1] == 0]
    num_people = len(detections)+19
    return num_people

def assign():
    lst = lst = ['butter milk - 10',
       'grape - 40',
       'lime - 20',
       'musk melon - 30',
       'water melon - 20',
       'orange - 70',
       'pineapple - 60',
       'musambi - 60',
       'pista milk - 20',
       'badam milkshake - 50',
       'pineapple drink - 20',
       'nutty bar - 30',
       'butterscotch cone - 35',
       'black current cone - 40',
       'black forest cake - 50',
       'idly(4) - 32',
       'wheat upma - 25',
       'chapathi egg masala - 93',
       'fish Meals - 75',
       'mathi fry - 50',
       'katla fry - 60',
       'single omelet - 15',
       'double omelet - 30',
       'egg briyani half - 50',
       'egg briyani - 80',
       'parotta - 90',
       'veg puff - 15',
       'badhusa - 15',
       'dill puff - 15',
       'bread toast - 15',
       'cookies - 5',
       'cutlet - 20',
       'Tea - 10',
       'coffee - 15',
       'boost - 35',
       'horlicks - 35',
       'bournvita - 35',
       'sukku coffee - 10',
       'black tea - 10'
    ]
    
    food = []
    cost = []
    for i in lst:
        food.append(i.split(" - ")[0])
        cost.append(i.split(" - ")[1])
    
    return food, cost

def predict(imgch):
    # Load TFLite model and allocate tensors.
    ms, me = 16, 23
    interpreter = tflite.Interpreter(
        model_path= r'C:\Users\Jegadit\Desktop\root\pah\works\python\AmritaCanteenApp\app\custom_model_lite\detect.tflite')
    # allocate the tensors
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_path = r'C:\Users\Jegadit\Desktop\root\pah\works\python\AmritaCanteenApp\Menu\_datasets\org\seq_4.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (320, 320))
    # Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_tensor = np.array(np.expand_dims(np.float32(img), 0))
    f, c = assign()
    for i in imgch:
        if i == 1:
            finf.append(f[:15])
            finc.append(c[:15])
        elif i == 2:
            finf.append(f[16:23])
            finc.append(c[16:23])
        elif i == 3:
            finf.append(f[24:])
            finc.append(c[24:])
    
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)

    return pred



app = Flask(__name__)
app.secret_key = "qwertuiop"
# app.permanent_session_lifetime = timedelta(minutes=1)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/menu')
def menu():
    predict([1,2,3])
    print(finf)
    return render_template('menu.html', fdata = flat(finf), cdata = flat(finc), size = len(flat(finf)))


@app.route('/ITstats')
def sides():
    return render_template('stats.html', count=detect())


if __name__ == '__main__':
    app.run(debug=True)
