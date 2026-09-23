import os
import json
import tempfile
import urllib.request
import zipfile
import numpy as np
import sherpa_onnx
import wave
from flask import Flask, request, jsonify, send_file
from gtts import gTTS

app = Flask(__name__)

# Разрешаем CORS, чтобы наш сайт мог достучаться
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'X-API-Key, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

API_KEY = os.environ.get("API_KEY", "plasti7154")

# ==== Настройка распознавателя ====
# Обрати внимание: мы не скачиваем модель в app.py, 
# мы просто указываем пути. Файлы должны быть скачаны build.sh
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder="encoder.chunk64.onnx",
    decoder="decoder.chunk64.onnx",
    joiner="joiner.chunk64.onnx",
    tokens="tokens.txt",
    num_threads=1, # На Render мало CPU, ставим 1
    sample_rate=16000,
    dither=3e-5,
    decoding_method="greedy_search",
    max_active_paths=10
)

def check_key():
    return request.headers.get("X-API-Key", "") == API_KEY

# ==== Функция чтения WAV (нужна для sherpa) ====
def read_wave(wave_filename):
    with wave.open(wave_filename, 'rb') as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()

@app.route("/stt", methods=["POST"])
def stt():
    if not check_key():
        return jsonify({"code": 2, "error": "Invalid API key"})
    if "audio" not in request.files:
        return jsonify({"code": 2, "error": "No audio file"})

    audio = request.files["audio"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.save(tmp.name)

    try:
        samples, sample_rate = read_wave(tmp.name)
        # Создаём поток для распознавания
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform=samples)
        # Добавляем паддинг в конце, чтобы модель «дослышала» последнее слово
        tail_padding = np.zeros(int(sample_rate * 0.5)).astype(np.float32)
        stream.accept_waveform(sample_rate, waveform=tail_padding)
        stream.input_finished()

        # Скармливаем данные и ждём результат
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        
        result = recognizer.get_result(stream)
    except Exception as e:
        os.unlink(tmp.name)
        return jsonify({"code": 2, "error": f"STT error: {str(e)}"})

    os.unlink(tmp.name)

    # Возвращаем результат с русскими буквами
    return app.response_class(
        json.dumps({"code": 1, "result": result.strip()}, ensure_ascii=False),
        mimetype="application/json; charset=utf-8"
    )

@app.route("/tts", methods=["POST"])
def tts():
    if not check_key():
        return jsonify({"code": 2, "error": "Invalid API key"})
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"code": 2, "error": "No text provided"})
    tts_obj = gTTS(text=text, lang="ru")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts_obj.save(tmp.name)
    return send_file(tmp.name, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"code": 1, "result": "Vosk STT/TTS API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
