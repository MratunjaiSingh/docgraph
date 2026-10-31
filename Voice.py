"""Voice helpers: speech-to-text via Groq Whisper, text-to-speech via the browser."""
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")   # more accurate: whisper-large-v3
MAX_SPEAK_CHARS = 1500
MIN_AUDIO_BYTES = 2000

_client = None


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """Audio (wav) -> text. language: 'en', 'hi' or None for auto-detect."""
    global _client
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        return ""
    if _client is None:
        _client = Groq()   # reads GROQ_API_KEY
    kwargs = {
        "file": ("question.wav", audio_bytes),
        "model": STT_MODEL,
        "response_format": "text",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    result = _client.audio.transcriptions.create(**kwargs)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return text.strip()


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # [label](url) -> label
    text = re.sub(r"https?://\S+", "link", text)               # don't read URLs aloud
    text = re.sub(r"[*_`#>|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_TEMPLATE = """
<div style="font-family:sans-serif;display:flex;gap:8px;align-items:center">
  <button id="play" style="padding:4px 12px;border-radius:6px;border:1px solid #bbb;cursor:pointer;background:#fff">🔊 Listen</button>
  <button id="stop" style="padding:4px 12px;border-radius:6px;border:1px solid #bbb;cursor:pointer;background:#fff">⏹ Stop</button>
  <span id="msg" style="font-size:12px;color:#b45309"></span>
</div>
<script>
  const text = __TEXT__;
  const lang = "__LANG__";
  const synth = window.speechSynthesis;
  let voices = synth.getVoices();
  synth.onvoiceschanged = () => { voices = synth.getVoices(); };
  const norm = v => v.lang.toLowerCase().replace("_", "-");
  function pickVoice() {
    const l = lang.toLowerCase();
    return voices.find(v => norm(v) === l) || voices.find(v => norm(v).startsWith(l.slice(0, 2)));
  }
  document.getElementById("play").onclick = () => {
    synth.cancel();
    const msg = document.getElementById("msg");
    msg.textContent = "";
    const voice = pickVoice();
    if (!voice && voices.length) {
      msg.textContent = "No " + (lang.startsWith("hi") ? "Hindi" : "English") + " voice installed on this device";
      return;
    }
    // speak sentence by sentence: Chrome cuts off long single utterances
    const parts = text.split(/(?<=[.!?\\u0964])\\s+/);
    parts.forEach(p => {
      const u = new SpeechSynthesisUtterance(p);
      u.lang = lang;
      if (voice) u.voice = voice;
      synth.speak(u);
    });
  };
  document.getElementById("stop").onclick = () => synth.cancel();
</script>
"""


def speak_button(text: str) -> None:
    """Render Listen/Stop buttons that read `text` aloud using the browser's built-in voices."""
    clean = _clean_for_speech(text)[:MAX_SPEAK_CHARS]
    if not clean:
        return
    letters = len(re.findall(r"\w", clean))
    devanagari = len(re.findall(r"[\u0900-\u097F]", clean))
    lang = "hi-IN" if letters and devanagari / letters > 0.3 else "en-IN"
    payload = json.dumps(clean).replace("<", "\\u003c")   # text is untrusted: never let it close the <script>
    html = _TEMPLATE.replace("__TEXT__", payload).replace("__LANG__", lang)
    if hasattr(st, "iframe"):          # st.components.v1.html is deprecated in recent Streamlit
        st.iframe(html, height=44)
    else:
        components.html(html, height=44)