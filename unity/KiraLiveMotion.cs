// 3.B: il personaggio chiede una posa nuova al bridge e la esegue, senza clip pre-cotte.
// Sul PC il bridge e' su localhost; sul visore basta puntare baseUrl all'IP del PC.
// Usa il client e il player del package (retarget muscle space, calibrazione root inclusa).

using AminHP.KimodoBridge;
using UnityEngine;

[RequireComponent(typeof(Animator))]
public class KiraLiveMotion : MonoBehaviour
{
    [Tooltip("Bridge: 127.0.0.1 sul PC, l'IP del PC quando gira sul visore.")]
    public string baseUrl = "http://127.0.0.1:8765";

    [Tooltip("Descrizione del movimento, in inglese come i prompt di motion capture.")]
    public string prompt = "a person waves hello with the right hand";

    public float durata = 3f;
    public int passiDiffusione = 100;

    private KimodoClient _client;
    private KimodoMotionPlayer _player;

    public string Stato { get; private set; } = "fermo";

    private void Awake()
    {
        _client = new KimodoClient(baseUrl) { GenerateTimeoutSeconds = 600 };
        _player = GetComponent<KimodoMotionPlayer>();
        if (_player == null) _player = gameObject.AddComponent<KimodoMotionPlayer>();
        _player.loop = false;
    }

    /// <summary>Chiede la posa al bridge; quando arriva la esegue.</summary>
    public void Chiedi(string testo = null, float? secondi = null)
    {
        if (_client == null) Awake();
        if (!string.IsNullOrEmpty(testo)) prompt = testo;
        if (secondi.HasValue) durata = secondi.Value;

        Stato = "genero: " + prompt;
        var richiesta = new KimodoGenerateRequest
        {
            prompt = prompt,
            model = "soma",
            duration = durata.ToString(System.Globalization.CultureInfo.InvariantCulture),
            num_samples = 1,
            diffusion_steps = passiDiffusione,
            postprocess = true,
            seed = -1,
        };
        _client.Generate(richiesta, (ok, motion, errore) =>
        {
            if (!ok || motion == null)
            {
                Stato = "errore: " + errore;
                Debug.LogError("[Kira] generazione fallita: " + errore);
                return;
            }
            bool partita = _player.Play(motion);
            Stato = partita
                ? $"eseguo {motion.frameCount} frame ({motion.generationSeconds:0.0}s di generazione)"
                : "posa ricevuta ma il player non e' partito";
            Debug.Log("[Kira] " + Stato);
        });
    }
}
