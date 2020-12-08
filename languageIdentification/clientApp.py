from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS, cross_origin

from kerasa.predict2 import predict
from com_in_ineuron_ai_utils.utils import decodeSound

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['sound']
    decodeSound(image, "inputSound.wav")
    result = predict()
    return jsonify({"Result" : result})


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=5000, debug=True)
