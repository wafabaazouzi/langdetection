import flask
import api_service as _api_service
from flask import request
from flask import abort
from flask import jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True





@app.route('/main/v1.0/Mergedfile', methods=['POST'])
def create_task():
    if not request.json or not 'text' in request.json:
        abort(400)
    txt = request.json['text']
    res = _api_service.predict(txt)
    return jsonify({'The language is in': res}), 201





#Run the API
app.run()

