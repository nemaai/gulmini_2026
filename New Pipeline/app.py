from flask import Flask
from routes.pipeline_routes import pipeline_bp

app = Flask(__name__)

app.register_blueprint(pipeline_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)