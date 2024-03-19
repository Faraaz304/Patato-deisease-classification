from flask import Flask , render_template , request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

class_name =['Early blight', 'Late blight', 'Healthy']

def reshape_image(image):
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    reshaped_image = np.expand_dims(image, axis=0)
    
    if image.shape[0] > 256 or image.shape[1] > 256:
        reshaped_image = tf.image.resize(reshaped_image, (256, 256))
    
    return reshaped_image


@app.route('/' , methods=['POST' , 'GET'] )
def home():
    if(request.method == 'POST'):
        im = request.files['img']
        im.save('static/img.jpg')
        im = cv2.imread('static/img.jpg')
        #print(im.shape)
        im  = reshape_image(im)
        #print(im.shape)
        
        



        model = tf.keras.models.load_model('model/model.h5')

        a =  model.predict(im)
        answer= class_name[(np.argmax(a))]
        
        
        return render_template('index.html',ans = answer)
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)