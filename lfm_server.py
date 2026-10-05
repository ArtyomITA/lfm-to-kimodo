# Servizio embedding LFM: POST /embed {"texts": [...]} -> {"vectors": [[...]]}
# stdlib + sentence-transformers. Percorsi e porta da variabili d'ambiente:
#   LFM_MODEL  cartella del modello LFM2.5-Embedding-350M (obbligatoria)
#   LFM_PORT   porta, default 8790
#   LFM_DEVICE cuda (default) oppure cpu
import json, os, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_DIR = os.environ.get("LFM_MODEL")
if not MODEL_DIR:
    sys.exit("LFM_MODEL non impostata: deve puntare alla cartella di LFM2.5-Embedding-350M")
PORT = int(os.environ.get("LFM_PORT", 8790))
DEVICE = os.environ.get("LFM_DEVICE", "cuda")

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm-server.log")
def log(m):
    with open(LOG, "a", encoding="ascii", errors="replace") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")
open(LOG, "w").close()

import torch
from sentence_transformers import SentenceTransformer
# trust_remote_code e fp32 non sono opzionali: con AutoModel liscio o in fp16 i vettori
# escono degeneri (varianza ~1e-12, cosine tutti 1.0).
model = SentenceTransformer(MODEL_DIR, trust_remote_code=True,
                            device=DEVICE, model_kwargs={"torch_dtype": torch.float32})
log(f"LFM pronto su :{PORT} ({DEVICE})")
print(f"LFM pronto su http://127.0.0.1:{PORT}/embed", flush=True)

class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def do_POST(s):
        n = int(s.headers.get("Content-Length", 0))
        req = json.loads(s.rfile.read(n))
        texts = req["texts"]
        v = model.encode(texts, prompt_name="document", batch_size=32,
                         convert_to_numpy=True, normalize_embeddings=False)
        body = json.dumps({"vectors": v.tolist()}).encode()
        s.send_response(200)
        s.send_header("Content-Type", "application/json")
        s.send_header("Content-Length", str(len(body)))
        s.end_headers()
        s.wfile.write(body)

HTTPServer(("127.0.0.1", PORT), H).serve_forever()
