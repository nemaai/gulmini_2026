from flask import Blueprint, request, jsonify
from services.pipeline_service import process_pipeline

pipeline_bp = Blueprint("pipeline", __name__)


@pipeline_bp.route("/api/pipeline/process", methods=["POST"])
def pipeline():

    if "file" not in request.files:
        return jsonify({"error": "No npy uploaded"}), 400

    file = request.files["file"]

    result = process_pipeline(file)

    return jsonify(result)