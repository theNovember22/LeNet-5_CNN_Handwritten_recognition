import json
import mimetypes
import os
from cgi import FieldStorage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model_utils import MODEL_PATH, load_trained_model, predict_digit


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

MODEL = None
MODEL_ERROR = None


def get_model():
    global MODEL, MODEL_ERROR
    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        MODEL_ERROR = (
            "Model file was not found. Run `python train_model.py --epochs 10` "
            "inside the CNN_as folder first."
        )
        return None

    try:
        MODEL = load_trained_model(MODEL_PATH)
    except Exception as exc:
        MODEL_ERROR = f"Could not load model: {exc}"
        return None
    return MODEL


class DigitAppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(TEMPLATE_DIR / "index.html", "text/html; charset=utf-8")
            return

        if self.path.startswith("/static/"):
            file_path = STATIC_DIR / self.path.removeprefix("/static/")
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            self._send_file(file_path, content_type)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        model = get_model()
        if model is None:
            self._send_json({"error": MODEL_ERROR}, status=503)
            return

        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        image_field = form["image"] if "image" in form else None

        if image_field is None or not image_field.file:
            self._send_json({"error": "Please upload an image file."}, status=400)
            return

        try:
            digit, confidence, probabilities, processed_image = predict_digit(model, image_field.file)
        except Exception as exc:
            self._send_json({"error": f"Prediction failed: {exc}"}, status=400)
            return

        self._send_json(
            {
                "digit": digit,
                "confidence": round(confidence * 100, 2),
                "probabilities": [round(value * 100, 2) for value in probabilities],
                "processedImage": processed_image,
            }
        )

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))

    def _send_file(self, file_path, content_type):
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "File not found"}, status=404)
            return

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), DigitAppHandler)
    print(f"Digit recognition app running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    run(host=host, port=port)
