#!/bin/bash
pip install -r requirements.txt

# Скачиваем стриминговую модель (вариант chunk64 для длинных записей)
wget https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main/am-onnx/encoder.chunk64.onnx
wget https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main/am-onnx/decoder.chunk64.onnx
wget https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main/am-onnx/joiner.chunk64.onnx
wget https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main/lang/tokens.txt
