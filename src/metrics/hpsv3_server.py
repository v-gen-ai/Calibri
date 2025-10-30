# scripts/hpsv3_server.py
import argparse
import io
import pickle
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from hpsv3 import HPSv3RewardInferencer

inferencer = None

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            body = self.rfile.read(length)
            payload = pickle.loads(body)
            prompts = payload["prompts"]
            jpeg_images = payload["images"]

            images = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpeg_images]
            torch.cuda.empty_cache()
            with torch.inference_mode():
                rewards = inferencer.reward(prompts=prompts, image_paths=images)
            scores = [reward[0].item() for reward in rewards]

            resp = pickle.dumps({"scores": scores})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            print("Error", str(e))
            msg = pickle.dumps({"error": str(e)})
            self.send_response(500)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18067)  # 18087
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global inferencer
    inferencer = HPSv3RewardInferencer(device=args.device)

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"HPSv3 server on http://{args.host}:{args.port}")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
