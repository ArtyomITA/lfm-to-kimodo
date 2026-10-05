# Da prompt ad animazione, con l'encoder LFM al posto di Llama-3-8B.
#
#   python genera.py "a person waves hello with the right hand" --durata 3 --gif
#   python genera.py --file prompts.txt --out animazioni
#
# Produce per ogni prompt:  <nome>.npz (formato Kimodo)  +  <nome>.bvh (scheletro, per
# Blender/Maya/Unity)  +  <nome>.gif con --gif.
#
# Serve lfm_server.py attivo (porta 8790 o LFM_URL).
# Da lanciare FUORI dal clone di kimodo, altrimenti l'import del pacchetto fallisce.

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(QUI, "bridge"))   # npc_dummy

LFM_URL = os.environ.get("LFM_URL", "http://127.0.0.1:8790/embed")
MAP_DIR = os.environ.get("MAP_DIR", os.path.join(QUI, "mappa"))
MODELLO = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1")
FPS = 30


def mappa_carica():
    W = np.load(os.path.join(MAP_DIR, "map_W_ridge.npy"))
    mu_x = np.load(os.path.join(MAP_DIR, "map_mu_x.npy"))
    mu_y = np.load(os.path.join(MAP_DIR, "map_mu_y.npy"))
    with open(os.path.join(MAP_DIR, "map_config.json")) as fh:
        cfg = json.load(fh)
    return W, mu_x, mu_y, cfg


def embed(testi, W, mu_x, mu_y, cfg):
    """LFM (1024) -> spazio che Kimodo si aspetta (4096)."""
    req = urllib.request.Request(LFM_URL, data=json.dumps({"texts": list(testi)}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            vettori = json.loads(r.read())["vectors"]
    except Exception as exc:
        raise SystemExit(f"lfm_server non raggiungibile su {LFM_URL}: {exc}")

    out = []
    for v in vettori:
        z = np.asarray(v, dtype=np.float32) - mu_x
        z = z / np.linalg.norm(z)
        y = z @ W
        y = y / np.linalg.norm(y)
        out.append((y * cfg["norm_y_media"] + mu_y).astype(np.float32))
    return out


def chiave(t):
    # sanitize_texts di kimodo ritocca il testo prima di passarlo all'encoder
    return t.strip().lower().rstrip(".")


class StubEncoder:
    """Serve gli embedding gia' calcolati. Zero pesi in memoria."""

    def __init__(self):
        self.tab = {}

    def add(self, testo, emb):
        self.tab[chiave(testo)] = emb

    def __call__(self, testi):
        import torch
        uno = isinstance(testi, str)
        lst = [testi] if uno else list(testi)
        arr = np.stack([self.tab[chiave(t)] for t in lst])[:, None, :]
        te = torch.tensor(arr)
        return (te[0], 1) if uno else (te, [1] * len(lst))

    def to(self, d):
        return self

    def eval(self):
        return self


def nome_file(prompt, i):
    base = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:48]
    return f"{i:02d}_{base}" if base else f"{i:02d}_motion"


def main():
    ap = argparse.ArgumentParser(description="prompt -> animazione (npz + bvh)")
    ap.add_argument("prompt", nargs="*", help='es: "a person waves hello with the right hand"')
    ap.add_argument("--file", help="file di testo con un prompt per riga")
    ap.add_argument("--durata", type=float, default=4.0, help="secondi per prompt (max 10)")
    ap.add_argument("--passi", type=int, default=100, help="passi di diffusione")
    ap.add_argument("--out", default="animazioni", help="cartella di uscita")
    ap.add_argument("--gif", action="store_true", help="salva anche un'anteprima gif")
    ap.add_argument("--tpose", action="store_true",
                    help="BVH con rest pose T standard invece di quella del dataset SEED")
    args = ap.parse_args()

    prompts = list(args.prompt)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            prompts += [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    if not prompts:
        ap.error('nessun prompt: passane uno fra virgolette oppure --file lista.txt')
    if args.durata > 10:
        print("nota: Kimodo genera al massimo 10 s per prompt, taglio a 10")
        args.durata = 10.0

    os.makedirs(args.out, exist_ok=True)

    print(f"embedding di {len(prompts)} prompt via LFM...")
    W, mu_x, mu_y, cfg = mappa_carica()
    embs = embed(prompts, W, mu_x, mu_y, cfg)

    # il preset dell'encoder va dirottato PRIMA di importare kimodo: altrimenti load_model
    # tira su LLM2Vec su Llama-3-8B (15 GB) che qui non serve a niente
    import importlib
    lm = importlib.import_module("kimodo.model.load_model")
    lm.TEXT_ENCODER_PRESETS["llm2vec"] = {"target": "npc_dummy.DummyEncoder", "kwargs": {}}

    import torch
    from kimodo import load_model
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.exports.motion_io import save_kimodo_npz

    print(f"carico {MODELLO}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(MODELLO, device=device, default_family="Kimodo",
                          return_resolved_name=True)
    stub = StubEncoder()
    model.text_encoder = stub

    n_frames = int(args.durata * FPS)
    for i, (prompt, emb) in enumerate(zip(prompts, embs)):
        stub.add(prompt, emb)
        t0 = time.time()
        with torch.no_grad():
            out = model([prompt], [n_frames], num_denoising_steps=args.passi,
                        num_samples=1, post_processing=True, return_numpy=True)

        # il modello restituisce un dizionario con la dimensione del batch davanti:
        # va tolta, altrimenti l'export non riconosce lo scheletro
        singolo = {k: (v[0] if hasattr(v, "shape") and v.shape[0] == 1 else v)
                   for k, v in out.items()}

        base = os.path.join(args.out, nome_file(prompt, i))
        save_kimodo_npz(base + ".npz", singolo)
        # il modello lavora internamente su somaskel30 ma emette 77 giunti: va passato
        # lo scheletro a 77, altrimenti l'export prova a riespandere quello che e' gia' espanso
        # ("value tensor of shape [T,77,3,3] cannot be broadcast to [T,30,3,3]")
        skel = getattr(model.skeleton, "somaskel77", model.skeleton)
        save_motion_bvh(base + ".bvh",
                        torch.as_tensor(singolo["local_rot_mats"]),
                        torch.as_tensor(singolo["root_positions"]),
                        skeleton=skel, fps=FPS, standard_tpose=args.tpose)
        print(f"  [{i}] {time.time() - t0:.1f}s  {prompt[:56]}")
        print(f"       {base}.npz  +  .bvh")

    if args.gif:
        sys.argv = ["npz_to_gif.py", args.out, "--out", args.out]
        sys.path.insert(0, os.path.join(QUI, "tools"))
        import npz_to_gif
        npz_to_gif.main()

    print("fatto:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
