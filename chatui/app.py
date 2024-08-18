from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import requests
import os

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# TRITON_SERVER_URL = "http://10.107.39.50:8000"  # Replace with your Triton server URL
Triton_server_url = os.environ.get("TRITON_SERVER_URL")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Prepare request to Triton Inference Server
    triton_request = {
        "inputs": [
            {
                "name": "prompt",  # Replace with actual input name
                "shape": [1],
                "datatype": "BYTES",  # Replace with actual data type
                "data": [user_input]
            }
        ],
        "outputs": [
            {
                "name": "generated_text"  # Replace with actual output name
            }
        ]
    }
    response = requests.post(f"{TRITON_SERVER_URL}/v2/models/llamav2/infer", json=triton_request)
    result = response.json()
    # Extract the necessary data from the Triton response
    model_output = result['outputs'][0]['data']
    model_output_str = model_output[0].split("[/INST]",1)[1]
    return jsonify({'response': model_output_str})

if __name__ == '__main__':
    app.run(debug=True)
