'''
Created on 
Course work: 
@author:
Source:
    https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-opencv
    https://www.geeksforgeeks.org/python-opencv-imdecode-function/
'''

from flask import *
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np


from flask import *
import cv2
import tensorflow as tf
import numpy as np


app = Flask(__name__)
new_size = (224, 224)

@app.route('/',methods=["GET","POST"])
def index():
    if request.method == "POST":
        img_data = request.files["input-image"].read()
        file_bytes = np.fromstring(img_data, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(img, new_size)
        resized_image = resized_image.reshape(-1 ,224,224,3)
        model = tf.keras.models.load_model('furniture.h5')
        pred = model.predict(resized_image)

        labels = {"0":"Bed","1":"Chair","2":"Sofa"}
        
        pred_label = str(np.where(pred[0]==1.0)[0][0])
        
        return render_template("index.html",result = labels[pred_label])
        
    
    return render_template("index.html")

if __name__== "__main__":
    app.run(host="0.0.0.0", debug = True, port = 8073)