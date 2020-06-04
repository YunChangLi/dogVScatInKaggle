# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions 
from keras.preprocessing import image 
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np 
import tkinter as tk
from tkinter import filedialog
#%matplotlib inline
#載入學習玩的模型VGG16
model = VGG16(weights='imagenet')


    
def predict(filename, featuresize):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=featuresize)[0]
    return results

def showimg(filename, title, i):
    
    im = Image.open(filename)
    print(im)
    im_list = np.asarray(im)
    plt.subplot(2, 5, i)
    plt.title(title)
    plt.axis("off")
    plt.imshow(im_list)

def getResult(filename):
    
    detail = []
    plt.figure(figsize=(20, 10))
    for i in range(1):
        showimg(filename, "query", i+1)
    plt.show()
    results = predict(filename, 10)
    for result in results:
        detail.append(str(result))
    
    ret = "\n".join(detail)
    print(ret)
    return ret
        
def Open():

    app.filename = filedialog.askopenfilename(initialdir="train", title="Select A Image")
    print(app.filename)
    text.insert('1.0' , getResult(app.filename))
    
app = tk.Tk()
app.geometry("600x600")
app.title("106502515")
button = tk.Button(app, text='Open', command=Open)
button.pack()
text = tk.Text(app)
text.pack()


app.mainloop()
    
