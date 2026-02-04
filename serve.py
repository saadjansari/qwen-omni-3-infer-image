"""
Flask server for SageMaker inference endpoint.
Handles /ping (health check) and /invocations (inference).
"""

import os
import sys
import logging
import boto3
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model reference
model = None
model_loaded = False


def get_hf_token():
    """Retrieve HF_TOKEN from Secrets Manager if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    
    try:
        secret_name = "huggingface-token"
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        
        # Handle both plain string and JSON formats
        secret = response["SecretString"]
        try:
            import json
            secret_dict = json.loads(secret)
            token = secret_dict["HF_TOKEN"]
        except (json.JSONDecodeError, KeyError):
            token = secret
        
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logger.info("HF_TOKEN loaded from Secrets Manager")
    except Exception as e:
        logger.warning(f"Could not load HF_TOKEN from Secrets Manager: {e}")


def load_model():
    """Load model on startup."""
    global model, model_loaded
    
    if model_loaded:
        return
    
    # Get HF token first
    get_hf_token()
    
    logger.info("Loading model...")
    
    # Import here to avoid loading before needed
    from inference import model_fn
    
    model = model_fn("/opt/ml/model")
    model_loaded = True
    
    logger.info("Model loaded successfully!")


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    # Load model on first ping if not loaded
    try:
        if not model_loaded:
            load_model()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """Inference endpoint."""
    try:
        # Ensure model is loaded
        if not model_loaded:
            load_model()
        
        # Parse input
        from inference import input_fn, predict_fn, output_fn
        
        content_type = request.content_type or "application/json"
        payload = input_fn(request.data.decode("utf-8"), content_type)
        
        # Run inference
        result = predict_fn(payload, model)
        
        # Format output
        response = output_fn(result, "application/json")
        
        return response, 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        import traceback
        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


if __name__ == "__main__":
    # Load model at startup
    logger.info("Starting server...")
    load_model()
    
    # Run server
    app.run(host="0.0.0.0", port=8080)
