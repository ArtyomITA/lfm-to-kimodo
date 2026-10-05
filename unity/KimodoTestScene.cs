// Costruisce la scena di prova: personaggio SMPL-X (rig Humanoid) + Animator con tutte le
// clip cotte dalla libreria, camera e luce. Uno stato per clip; si passa da una all'altra
// col trigger che porta il nome della posa (es. "wave_right").

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class KimodoTestScene
{
    const string ClipsDir = "Assets/KimodoClips";
    const string ScenePath = "Assets/Scenes/KimodoTest.unity";
    const string ControllerPath = "Assets/KimodoClips/KimodoLibrary.controller";
    const string ModelPath = "Assets/SMPLX/Models/smplx-neutral.fbx";
    const string PosaIniziale = "idle";

    [MenuItem("Kimodo/Crea scena di prova")]
    public static void Create() { Debug.Log(CreateReport()); }

    public static string CreateReport()
    {
        var clips = new List<AnimationClip>();
        foreach (var f in Directory.GetFiles(ClipsDir, "*.anim"))
        {
            var c = AssetDatabase.LoadAssetAtPath<AnimationClip>(f.Replace(System.IO.Path.DirectorySeparatorChar, '/'));
            if (c != null) clips.Add(c);
        }
        if (clips.Count == 0) return "[Kimodo] nessuna clip in " + ClipsDir;

        // --- Animator controller: uno stato per clip, trigger omonimo ---
        AssetDatabase.DeleteAsset(ControllerPath);
        var ctrl = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
        var sm = ctrl.layers[0].stateMachine;
        var stati = new Dictionary<string, AnimatorState>();

        foreach (var c in clips)
        {
            ctrl.AddParameter(c.name, AnimatorControllerParameterType.Trigger);
            var st = sm.AddState(c.name);
            st.motion = c;
            stati[c.name] = st;
            if (c.name == PosaIniziale) sm.defaultState = st;
        }
        // da ogni stato si puo' saltare a ogni altro col suo trigger
        foreach (var da in stati.Values)
            foreach (var kv in stati)
            {
                if (kv.Value == da) continue;
                var tr = da.AddTransition(kv.Value);
                tr.AddCondition(AnimatorConditionMode.If, 0f, kv.Key);
                tr.hasExitTime = false;
                tr.duration = 0.15f;
            }

        // --- scena ---
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var luce = new GameObject("Sole");
        var l = luce.AddComponent<Light>();
        l.type = LightType.Directional; l.intensity = 1.1f; l.shadows = LightShadows.Soft;
        luce.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

        var camGo = new GameObject("Camera");
        var cam = camGo.AddComponent<Camera>();
        camGo.tag = "MainCamera";
        camGo.transform.position = new Vector3(0f, 1.1f, 3.2f);
        camGo.transform.rotation = Quaternion.Euler(3f, 180f, 0f);
        cam.backgroundColor = new Color(0.16f, 0.17f, 0.20f);
        cam.clearFlags = CameraClearFlags.SolidColor;

        var piano = GameObject.CreatePrimitive(PrimitiveType.Plane);
        piano.name = "Pavimento";
        piano.transform.localScale = new Vector3(2f, 1f, 2f);

        var modello = AssetDatabase.LoadAssetAtPath<GameObject>(ModelPath);
        if (modello == null) return "[Kimodo] modello non trovato: " + ModelPath;
        var attore = (GameObject)PrefabUtility.InstantiatePrefab(modello);
        attore.name = "Kira";
        attore.transform.position = Vector3.zero;

        var anim = attore.GetComponent<Animator>();
        if (anim == null) anim = attore.AddComponent<Animator>();
        anim.runtimeAnimatorController = ctrl;
        anim.applyRootMotion = false;

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        return $"[Kimodo] scena {ScenePath} creata: {clips.Count} clip nel controller, "
             + $"attore='{attore.name}' avatar={(anim.avatar != null ? anim.avatar.name : "NULL")} "
             + $"isHuman={(anim.avatar != null && anim.avatar.isHuman)}, stato iniziale='{PosaIniziale}'";
    }
}
