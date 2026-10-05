# Anteprima GIF delle pose generate: scheletro proiettato + prompt che l'ha prodotta.
# Solo PIL (gia' nei nostri env), niente Unity acceso, niente dipendenze nuove.
#
# uso:  python npz_to_gif.py <cartella_o_npz ...> [--out CARTELLA]
# env:  KIMODO_REPO   clone di kimodo (serve per le connessioni fra le ossa); se assente
#                     cerca nel pacchetto kimodo installato
#       MOTION_INDEX  index.json opzionale {nome: {intent: "..."}} per stampare il prompt

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REL_SKIN = os.path.join("kimodo", "assets", "skeletons", "somaskel77", "skin_standard.npz")
INDEX = os.environ.get("MOTION_INDEX", "")
OUT = os.path.join(os.getcwd(), "gif")
W = H = 480
FPS_OUT = 15          # 30 fps sorgente, si tiene un frame su due: meta' peso, movimento uguale
SFONDO = (26, 28, 33)
OSSO = (150, 200, 255)
GIUNTO = (255, 255, 255)
TESTO = (235, 235, 235)


def trova_skin():
    """skin_standard.npz: nel clone di kimodo oppure nel pacchetto installato."""
    repo = os.environ.get("KIMODO_REPO")
    if repo:
        for base in (repo, os.path.join(repo, "kimodo")):
            p = os.path.join(base, REL_SKIN)
            if os.path.exists(p):
                return p
    try:
        import kimodo
        p = os.path.join(os.path.dirname(kimodo.__file__), "assets", "skeletons",
                         "somaskel77", "skin_standard.npz")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    raise SystemExit("skin_standard.npz non trovato: imposta KIMODO_REPO al clone di kimodo")


def archi():
    d = np.load(trova_skin(), allow_pickle=True)
    return d["rig_joint_connections"].astype(int)


def prompts():
    if not INDEX or not os.path.exists(INDEX):
        return {}
    with open(INDEX, encoding="utf-8") as fh:
        return {k: v.get("intent", "") for k, v in json.load(fh).items()}


def a_pixel(punti, minimi, scala):
    """Proiezione ortografica sul piano x-y (Kimodo: y su, z avanti), origine in basso al centro."""
    xy = (punti[:, :2] - minimi) * scala
    px = xy[:, 0] + (W - (xy[:, 0].max() - xy[:, 0].min())) / 2 - xy[:, 0].min()
    py = H - 40 - xy[:, 1]
    return np.stack([px, py], axis=1)


def disegna(npz_path, testo, salta=2):
    d = np.load(npz_path, allow_pickle=True)
    if "posed_joints" not in d:
        raise ValueError(f"{os.path.basename(npz_path)}: manca posed_joints")
    pose = np.asarray(d["posed_joints"], np.float32)[::salta]   # (T, J, 3)

    # scala e inquadratura fisse su tutta la clip: il personaggio non "respira" tra un frame e l'altro
    piatto = pose.reshape(-1, 3)
    minimi = piatto[:, :2].min(axis=0)
    span = (piatto[:, :2].max(axis=0) - minimi).max()
    scala = (H - 110) / max(span, 1e-6)

    coppie = archi()
    frames = []
    for giunti in pose:
        img = Image.new("RGB", (W, H), SFONDO)
        dr = ImageDraw.Draw(img)
        p = a_pixel(giunti, minimi, scala)
        for a, b in coppie:
            dr.line([tuple(p[a]), tuple(p[b])], fill=OSSO, width=3)
        for x, y in p:
            dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=GIUNTO)
        dr.text((12, 10), testo[:62], fill=TESTO)
        if len(testo) > 62:
            dr.text((12, 24), testo[62:124], fill=TESTO)
        frames.append(img)
    return frames


def main():
    args = list(sys.argv[1:])
    out_dir = OUT
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        del args[i:i + 2]
    if not args:
        raise SystemExit("uso: npz_to_gif.py <cartella_o_npz ...> [--out CARTELLA]")
    files = []
    for a in args:
        if os.path.isdir(a):
            files += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.endswith(".npz")]
        elif a.endswith(".npz"):
            files.append(a)

    os.makedirs(out_dir, exist_ok=True)
    testi = prompts()
    durata_ms = int(1000 / FPS_OUT)

    for f in files:
        nome = os.path.splitext(os.path.basename(f))[0]
        testo = testi.get(nome, nome)
        try:
            frames = disegna(f, f"{nome}: {testo}")
        except Exception as exc:
            print(f"  SALTATO {nome}: {exc}")
            continue
        out = os.path.join(out_dir, nome + ".gif")
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=durata_ms, loop=0, optimize=True)
        print(f"  {nome}: {len(frames)} frame -> {os.path.getsize(out)/1e6:.1f} MB")

    print("in", out_dir)


if __name__ == "__main__":
    main()
