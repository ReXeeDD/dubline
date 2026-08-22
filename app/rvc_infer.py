"""Persistent RVC conversion worker.

Runs under the Applio virtualenv, not the app's interpreter - Applio needs
torch+CUDA and its own pinned dependency set, which must not be mixed into the
dubbing app's environment.

Loading an RVC model costs ~14 seconds, which is far more than converting a
chunk of audio. So this stays alive and processes jobs from stdin, keeping the
model and the hubert embedder resident: one load per session instead of one per
chunk. Applio's VoiceConverter reloads only when the weight path changes, so
repeated calls are effectively free.

Applio's converter prints freely to stdout, so every reply is tagged with a
sentinel prefix and the caller ignores everything else.

Protocol: one JSON object per line on stdin, one tagged JSON line back per job.
    -> {"in": "a.wav", "out": "b.wav", "pth": "...", "index": "...",
        "index_rate": 0.5, "protect": 0.33, "soften": 35}
    <- {"ok": true}  |  {"ok": false, "error": "..."}
Send an empty line or close stdin to exit.
"""
from __future__ import annotations

import json
import os
import sys


TAG = "##RVC##"


def main() -> int:
    applio = sys.argv[1]
    os.chdir(applio)
    sys.path.insert(0, applio)

    from rvc.infer.infer import VoiceConverter

    def reply(obj):
        print(TAG + json.dumps(obj), flush=True)

    vc = VoiceConverter()
    loaded = None

    # Tell the caller we are up; model load happens on the first job.
    reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        try:
            job = json.loads(line)
        except Exception as e:
            reply({"ok": False, "error": f"bad job: {e}"})
            continue

        try:
            pth = job["pth"]
            # Softness maps onto the spectral cleaner. The relationship is
            # measured, not guessed: strength 0.3 leaves speaker match at 0.983,
            # 0.7 drops it to 0.976. 0.8 is the ceiling the UI allows.
            soften = min(0.8, max(0.0, float(job.get("soften", 0)) / 100.0 * 0.8))
            if loaded != pth:
                vc.get_vc(pth, 0)
                loaded = pth

            vc.convert_audio(
                audio_input_path=job["in"],
                audio_output_path=job["out"],
                model_path=pth,
                index_path=job.get("index") or "",
                pitch=0,                      # pitch is applied upstream by edge-tts
                f0_method="rmvpe",
                index_rate=float(job.get("index_rate", 0.5)),
                volume_envelope=1.0,
                protect=float(job.get("protect", 0.33)),
                # Whole-file conversion. Applio's splitter re-times the audio and
                # loses a fraction of a second per pass, which would compound into
                # seconds of drift across an episode; the caller restores exact
                # length and re-gates the silence instead.
                split_audio=False,
                f0_autotune=False,
                clean_audio=soften > 0,
                clean_strength=soften,
                export_format="WAV",
                embedder_model="contentvec",
                sid=0,
            )
            ok = os.path.exists(job["out"])
            reply({"ok": True} if ok else
                  {"ok": False, "error": "no output written"})
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
