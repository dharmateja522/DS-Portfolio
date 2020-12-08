import numpy as np
import os
import sys
from keras.models import load_model

#from data_loaders.SpectrogramGenerator import SpectrogramGenerator
from kerasa.data_loaders.SpectrogramGenerator import SpectrogramGenerator

class_labels = ["ENGLISH", "GERMAN", "FRENCH", "Espanol", "Chinese", "RUSSIAN"]



def predict():
    inputfile = "inputSound.wav"
    config = {"pixel_per_second": 50, "input_shape": [129, 500, 1], "num_classes": 4}
    data_generator = SpectrogramGenerator(inputfile, config, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # Model Generation
    model = load_model("2017-01-31-14-29-14.CRNN_EN_DE_FR_ES_CN_RU.model")

    probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    #print(classes, class_labels[average_class], average_prob)
    return class_labels[average_class]

#if __name__ == "__main__":

#predict()