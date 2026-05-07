import json
import flask
import sys
import os

sys.path.insert(0, "/opt/ml/code")
import inference

app = flask.Flask(__name__)
model_artifacts = None

def get_model():
    global model_artifacts
    if model_artifacts is None:
        model_artifacts = inference.model_fn("/opt/ml/model")
    return model_artifacts

@app.route("/ping", methods=["GET"])
def ping():
    try:
        get_model()
        return flask.Response(response="{}", status=200, mimetype="application/json")
    except Exception as e:
        return flask.Response(response=json.dumps({"error": str(e)}), status=500, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def invoke():
    content_type = flask.request.content_type or "application/json"
    body = flask.request.data.decode("utf-8")
    input_data = inference.input_fn(body, content_type)
    prediction = inference.predict_fn(input_data, get_model())
    output, output_type = inference.output_fn(prediction, "application/json")
    return flask.Response(response=output, status=200, mimetype=output_type)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)