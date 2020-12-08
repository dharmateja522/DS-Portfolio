from wsgiref import simple_server
from flask import Flask, request, jsonify
from flask import Response
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_predict.predict import LogoClassification
from com_in_ineuron_ai_utils.utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class Api:
    def __init__(self):
        self.classifier = LogoClassification()


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, "inputImage.jpg")
        result = cliApp.classifier.getPrediction("inputImage.jpg")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


port = int(os.getenv("PORT"))
if __name__ == "__main__":
    cliApp = Api()
    host = '0.0.0.0'
    #port = 6000
    httpd = simple_server.make_server(host, port, app)
    #print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
