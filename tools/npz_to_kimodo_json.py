# Converte gli npz generati da Kimodo (SOMA 77 joint) nel JSON che il package Unity
# com.aminhp.kimodobridge si aspetta dalla risposta /generate del bridge server.
# Cosi' le pose gia' generate entrano in Unity senza rigenerare nulla e senza server acceso.
#
# uso:  python npz_to_kimodo_json.py <cartella_o_file ...> [--out CARTELLA]
# out:  default <cartella corrente>\KimodoMotions\<nome>.json, oppure MOTION_JSON_OUT
#
# Da lanciare FUORI dal clone di kimodo: li' dentro la cartella kimodo/ ombreggia il
# pacchetto installato e l'import muore con "cannot import name 'skeleton_asset_path'".

import json
import os
import sys

import numpy as np

OUT_DIR = os.environ.get("MOTION_JSON_OUT", os.path.join(os.getcwd(), "KimodoMotions"))


def mats_to_quats_wxyz(m):
    """[..., 3, 3] -> [..., 4] wxyz. Shepperd: sceglie il ramo col denominatore piu' grande."""
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)

    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    big_w = t > 0
    i = np.argmax(np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1), axis=1)

    # ramo w
    idx = np.where(big_w)[0]
    if idx.size:
        s = np.sqrt(t[idx] + 1.0) * 2.0
        q[idx, 0] = 0.25 * s
        q[idx, 1] = (m[idx, 2, 1] - m[idx, 1, 2]) / s
        q[idx, 2] = (m[idx, 0, 2] - m[idx, 2, 0]) / s
        q[idx, 3] = (m[idx, 1, 0] - m[idx, 0, 1]) / s

    # rami x/y/z (traccia <= 0): permutazione ciclica
    for axis in (0, 1, 2):
        idx = np.where((~big_w) & (i == axis))[0]
        if not idx.size:
            continue
        a, b, c = axis, (axis + 1) % 3, (axis + 2) % 3
        s = np.sqrt(1.0 + m[idx, a, a] - m[idx, b, b] - m[idx, c, c]) * 2.0
        q[idx, 0] = (m[idx, c, b] - m[idx, b, c]) / s
        q[idx, 1 + a] = 0.25 * s
        q[idx, 1 + b] = (m[idx, b, a] + m[idx, a, b]) / s
        q[idx, 1 + c] = (m[idx, c, a] + m[idx, a, c]) / s

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # segno canonico: w >= 0 (stessa rotazione, evita salti tra frame vicini)
    flip = q[:, 0] < 0
    q[flip] *= -1.0
    return q.reshape(*shape, 4).astype(np.float32)


def load_skeleton():
    """Lo scheletro di export (somaskel77), lo stesso che usa il bridge server."""
    from kimodo.skeleton import SOMASkeleton77
    return SOMASkeleton77()


def skeleton_bones():
    """Stessa lista ossa del bridge server: nome, parent, offset locale rest (metri, coord Kimodo)."""
    skel = load_skeleton()
    neutral = skel.neutral_joints.detach().cpu().numpy()
    parents = skel.joint_parents.detach().cpu().numpy().astype(int)
    names = list(skel.bone_order_names)

    # neutral_joints ha la radice nell'origine: un rig costruito cosi' ha l'anca a y=0 e
    # Unity ne deduce un personaggio alto pochi millimetri (humanScale ~0.009), per cui
    # HumanPoseHandler restituisce bodyPosition ~ -110 m e il retarget spara il modello
    # sottoterra. Si alza la radice all'altezza reale dell'anca (piedi a y=0): humanScale
    # torna ~1 e la traslazione e' corretta. Le rotazioni non cambiano.
    altezza_anca = float(-neutral[:, 1].min())

    bones = []
    for idx, name in enumerate(names):
        p = int(parents[idx])
        off = neutral[idx] if p < 0 else (neutral[idx] - neutral[p])
        if p < 0:
            off = np.array([off[0], altezza_anca, off[2]], dtype=np.float32)
        bones.append({"name": name, "parent": p,
                      "ox": float(off[0]), "oy": float(off[1]), "oz": float(off[2])})
    root = int(np.where(parents < 0)[0][0])
    return bones, root, names


def convert(npz_path, bones, root_index, names, skel, fps=30.0):
    d = np.load(npz_path, allow_pickle=True)
    if "local_rot_mats" not in d:
        raise ValueError(f"{os.path.basename(npz_path)}: manca local_rot_mats (chiavi: {list(d.files)})")

    import torch
    from kimodo.skeleton import global_rots_to_local_rots

    # Stessa strada del bridge server: si parte dalle rotazioni GLOBALI e le si converte
    # in locali sullo scheletro di export. Le local_rot_mats dell'npz sono relative allo
    # scheletro interno del modello e su un rig a 77 ossa danno pose sbagliate.
    g_rot = d["global_rot_mats"]                    # (T, J, 3, 3)
    if g_rot.ndim != 4 or g_rot.shape[-2:] != (3, 3):
        raise ValueError(f"{os.path.basename(npz_path)}: global_rot_mats ha forma {g_rot.shape}, attesa (T,J,3,3)")
    frames, joints = g_rot.shape[0], g_rot.shape[1]
    if joints != len(bones):
        raise ValueError(f"{os.path.basename(npz_path)}: {joints} joint, lo scheletro ne ha {len(bones)}")

    local_rot = global_rots_to_local_rots(torch.from_numpy(np.asarray(g_rot, np.float32)), skel)
    quats = mats_to_quats_wxyz(local_rot.detach().cpu().numpy())   # (T, J, 4) wxyz

    # root = posizione del joint radice nelle pose, come fa il server (non root_positions)
    if "posed_joints" in d:
        root_pos = np.asarray(d["posed_joints"], np.float32)[:, root_index, :]
    else:
        root_pos = d["root_positions"] if "root_positions" in d else np.zeros((frames, 3), np.float32)
    foot = d["foot_contacts"].astype(np.float32) if "foot_contacts" in d else np.zeros((frames, 0), np.float32)

    motion = {
        "model": "kimodo-soma-rp",
        "displayName": "Kimodo SOMA RP v1.1 (offline npz)",
        "skeletonName": "somaskel77",
        "coordSystem": "kimodo",
        "quatOrder": "wxyz",
        "fps": float(fps),
        "frameCount": int(frames),
        "jointCount": int(joints),
        "rootIndex": int(root_index),
        "footContactChannels": int(foot.shape[1]),
        "bones": bones,
        "clips": [{
            "rootPositions": np.asarray(root_pos, np.float32).reshape(-1).tolist(),
            "localQuats": quats.reshape(-1).tolist(),
            "footContacts": foot.reshape(-1).tolist(),
            "posedJoints": [],
        }],
        "prompts": [os.path.splitext(os.path.basename(npz_path))[0]],
        "numFramesPerSegment": [int(frames)],
        "generationSeconds": 0.0,
        "detail": "",
    }
    return motion, frames, joints


def main():
    args = list(sys.argv[1:])
    out_dir = OUT_DIR
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        del args[i:i + 2]
    if not args:
        raise SystemExit("uso: npz_to_kimodo_json.py <cartella_o_file ...> [--out CARTELLA]")
    files = []
    for a in args:
        if os.path.isdir(a):
            files += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.endswith(".npz")]
        elif a.endswith(".npz"):
            files.append(a)
    if not files:
        print("nessun npz trovato in:", args)
        return 1

    skel = load_skeleton()
    bones, root_index, names = skeleton_bones()
    os.makedirs(out_dir, exist_ok=True)
    print(f"scheletro: {len(bones)} ossa, root={names[root_index]}")

    ok, ko = 0, 0
    for f in files:
        nome = os.path.splitext(os.path.basename(f))[0]
        try:
            motion, frames, joints = convert(f, bones, root_index, names, skel)
        except Exception as exc:
            print(f"  SALTATO {nome}: {exc}")
            ko += 1
            continue
        out = os.path.join(out_dir, nome + ".json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(motion, fh)
        mb = os.path.getsize(out) / 1e6
        print(f"  {nome}: {frames} frame x {joints} joint -> {mb:.1f} MB")
        ok += 1

    print(f"fatti {ok}, saltati {ko}, in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
