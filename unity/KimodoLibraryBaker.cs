// Importa i JSON prodotti da tools/npz_to_kimodo_json.py e li cuoce in AnimationClip
// Humanoid, riusando il retarget muscle-space del package com.aminhp.kimodobridge.
// Nessun server acceso: le pose sono gia' generate, qui si converte e basta.

using System.Collections.Generic;
using System.IO;
using AminHP.KimodoBridge;
using AminHP.KimodoBridge.Editor;
using UnityEditor;
using UnityEngine;

public static class KimodoLibraryBaker
{
    const string MotionsDir = "Assets/KimodoMotions";
    const string ClipsDir = "Assets/KimodoClips";

    [MenuItem("Kimodo/Bake libreria pose (JSON -> AnimationClip)")]
    public static void BakeAll()
    {
        var report = BakeAllReport(loopIdle: true);
        Debug.Log(report);
    }

    /// <summary>Cuoce ogni JSON in una clip. Ritorna un report testuale (usabile da eval).</summary>
    public static string BakeAllReport(bool loopIdle)
    {
        if (!Directory.Exists(MotionsDir))
            return $"[Kimodo] cartella mancante: {MotionsDir}";

        Directory.CreateDirectory(ClipsDir);
        var files = Directory.GetFiles(MotionsDir, "*.json");
        System.Array.Sort(files);

        var righe = new List<string>();
        int ok = 0, ko = 0;

        foreach (var file in files)
        {
            string nome = Path.GetFileNameWithoutExtension(file);
            try
            {
                var motion = JsonUtility.FromJson<KimodoMotion>(File.ReadAllText(file));
                if (motion == null || motion.bones == null || motion.bones.Count == 0)
                { righe.Add($"  {nome}: JSON senza ossa"); ko++; continue; }

                // loop solo per le pose cicliche; le altre finiscono e basta.
                bool loop = loopIdle && (nome.Contains("idle") || nome.Contains("walk") || nome.Contains("run"));
                // travel vero per le locomozioni, sul posto per il resto.
                var rootBake = (nome.Contains("walk") || nome.Contains("run"))
                    ? KimodoRootBake.Travel : KimodoRootBake.InPlace;

                string path = ClipsDir + "/" + nome + ".anim";
                if (File.Exists(path)) AssetDatabase.DeleteAsset(path);
                string creato = KimodoBaker.BakeToAsset(motion, 0, loop, path, 1f, rootBake);

                var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(creato);
                righe.Add($"  {nome}: {motion.frameCount} frame, {clip.length:0.00}s, loop={loop}, root={rootBake} -> {creato}");
                ok++;
            }
            catch (System.Exception e)
            {
                righe.Add($"  {nome}: ERRORE {e.GetType().Name}: {e.Message}");
                ko++;
            }
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        return $"[Kimodo] bake: {ok} clip create, {ko} errori\n" + string.Join("\n", righe);
    }
}
