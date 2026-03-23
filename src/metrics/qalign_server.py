# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.5.1+cu118",
#     "torchvision==0.20.1+cu118",
#     "pillow",
#     "transformers==4.36.1",
#     "accelerate",
#     "sentencepiece",
#     "requests",
#     "icecream",
#     "protobuf",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu118" }
# torchvision = { index = "pytorch-cu118" }
# torchaudio = { index = "pytorch-cu118" }
#
# [[tool.uv.index]]
# name = "pytorch-cu118"
# url = "https://download.pytorch.org/whl/cu118"
# explicit = true
# ///

# scripts/qalign_server.py
import argparse
import io
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

model = None
default_task = "quality"     # quality | aesthetics
default_input = "image"      # image | video

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = pickle.loads(body)

            jpeg_images = payload.get("images", [])
            task = payload.get("task", default_task)
            input_type = payload.get("input", default_input)

            # prompts могут присутствовать ради совместимости, но не используются
            # prompts = payload.get("prompts", None)

            images = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpeg_images]

            torch.cuda.empty_cache()
            with torch.inference_mode():
                out = model.score(images, task_=task, input_=input_type)

            # Приведение к списку float
            try:
                if isinstance(out, torch.Tensor):
                    scores = [float(x) for x in out.detach().cpu().flatten().tolist()]
                elif hasattr(out, "tolist"):
                    scores = [float(x) for x in out.tolist()]
                else:
                    scores = [float(x) for x in out]
            except Exception:
                scores = [float(out)] if not isinstance(out, (list, tuple)) else [float(x) for x in out]

            resp = pickle.dumps({"scores": scores})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            msg = pickle.dumps({"error": str(e)})
            self.send_response(500)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18088)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_id", default="q-future/one-align")
    args = parser.parse_args()

    global model
    # Соответствует документации One-Align: trust_remote_code=True, fp16, device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="eager",  # как в Quick Start
    ).eval()

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Q-Align server on http://{args.host}:{args.port}")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
