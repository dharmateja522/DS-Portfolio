import numpy as np
from itertools import chain
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2


class LogoClassification:
    def __init__(self):
        # self.class_names = ['backpack', 'earring', 'footwear', 'GirlsTop', 'glasses', 'Jacket', 'LadiesHandbag',
        #                    'mensShorts', 'MensTopWear', 'saree', 'trousers', 'wallet', 'watch']
        # self.class_names = ['Apple', 'barcelona', 'CocaCola', 'ebay', 'ford', 'Google', 'kfc',
        #                    'MacDonald', 'Mercedes', 'Nike', 'Shell', 'Starbucks']
        #self.class_names = ['Aeroplane', 'Auto', 'Bicycle', 'Bike', 'Boat', 'Bus', 'Car',
        #                    'Scooty', 'Ship', 'Train', 'Truck']
        #self.class_names = ['rock', 'paper', 'scissor']
        self.class_names = ['Goggles','Hat','Jacket','Shirt','Shoes','Shorts','T-Shirt','Trouser','Wallet','Watch']
        self.model = load_model("models/fashion.h5")

    def getPrediction(self, img):
        img = cv2.imread(img)
        dim = (224, 224)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        print(preds)
        preds_unlist = list(chain(*preds))
        print(preds_unlist)
        preds_int = [int((round(i, 2))) for i in preds_unlist]
        print(preds_int)
        # self.final_pred_unused = dict(zip(self.class_names,self.preds_int))
        final_pred = dict(zip(self.class_names, preds_int))
        # finale = final_pred[1]
        print(100 * '-')
        print(final_pred)
        return final_pred
