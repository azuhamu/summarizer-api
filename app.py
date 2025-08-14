import os, re, gc, threading
from functools import lru_cache
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# メモリ・並列抑制
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", "/opt/render/project/.cache/huggingface")

MODEL_ID = os.getenv("MODEL_ID", "t5-small")  # 例: sshleifer/tiny-t5（検証用に軽量）
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1000"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "140"))

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024  # 512KB

_lock = threading.Lock()
_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return
    with _lock:
        if _model is not None and _tokenizer is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
        _model = T5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True
        )
        _model.eval()

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:MAX_INPUT_CHARS]

def _trim_140(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:MAX_SUMMARY_CHARS]

def _extractive_fallback(text: str) -> str:
    parts = [p for p in re.split(r"[。．!?！？]+", text) if p]
    out = ""
    for p in parts:
        candidate = (out + ("。" if out else "") + p).strip()
        if len(candidate) > MAX_SUMMARY_CHARS:
            break
        out = candidate
    if not out:
        out = text[:MAX_SUMMARY_CHARS]
    return _trim_140(out)

@lru_cache(maxsize=128)
def _summarize_core(text: str) -> str:
    import torch
    _load_model()
    prefix = "summarize: "
    input_text = prefix + text
    with torch.no_grad():
        input_ids = _tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        output_ids = _model.generate(
            input_ids,
            max_length=96,
            min_length=24,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    del input_ids, output_ids
    gc.collect()
    return _trim_140(summary)

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/", methods=["GET"])
def root():
    return "summarizer alive", 200

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True, silent=False)
        text = (data.get("text") or "").strip()
        if not text:
            raise BadRequest("Missing or empty 'text' field.")
        text = _clean_text(text)
        try:
            result = _summarize_core(text)
        except RuntimeError:
            result = _extractive_fallback(text)
        return jsonify({"summary": result})
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
