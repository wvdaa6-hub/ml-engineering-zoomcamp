from flask import Flask, jsonify, request
from predict import load_model, predict


app = Flask("water-potability")

# Load the model once when the service starts, not on every request
model = load_model()


# Simple check that the service is running
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# Receive one water sample as JSON and return the prediction
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    sample = request.get_json()
    if not sample:
        return jsonify({"error": "Send a JSON object with the water measurements"}), 400
    return jsonify(predict(model, sample))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)