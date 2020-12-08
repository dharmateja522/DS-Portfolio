from wsgiref import simple_server
from flask import Flask, request, jsonify
from flask import Response
import os
from flask_cors import CORS
from kerasa.predict2 import predict
from com_in_ineuron_ai_utils.utils import decodeSound
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        image = request.json['sound']
        decodeSound(image, "audio123.wav")
        result = predict()
        #print(result)
        #print(type(result))
        #return jsonify(result)

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return {"Result" : result}


#port = int(os.getenv("PORT"))

if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()