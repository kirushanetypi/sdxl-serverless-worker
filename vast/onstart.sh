#!/bin/bash
# Onstart Vast-инстанса для образа comfyui-canon: ComfyUI уже внутри образа
# вместе с чекпойнтом канона, докачивать на инстансе нечего.
#
# Скрипт идемпотентный: если ComfyUI из CMD образа уже слушает 8188 — просто
# дожидается готовности, второй процесс не поднимает (иначе два main.py
# подерутся за порт).
set -x
LOG=/workspace/onstart.log
mkdir -p /workspace
exec >>"$LOG" 2>&1

echo "[onstart] start $(date -u)"
nvidia-smi || echo "[onstart] nvidia-smi недоступен"

COMFY_DIR=/opt/ComfyUI
CKPT="$COMFY_DIR/models/checkpoints/waiNSFWIllustrious_v150.safetensors"

if [ ! -f "$CKPT" ]; then
    echo "[onstart] СТОП: в образе нет $CKPT — это не наш comfyui-canon"
    exit 1
fi
echo "[onstart] чекпойнт на месте: $(stat -c %s "$CKPT") байт"

ready() { curl -sf http://localhost:8188/system_stats >/dev/null 2>&1; }

if ready; then
    echo "[onstart] ComfyUI из CMD образа уже отвечает"
else
    echo "[onstart] запускаю ComfyUI"
    cd "$COMFY_DIR" || { echo "[onstart] нет каталога $COMFY_DIR"; exit 1; }
    nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch > /workspace/comfyui.log 2>&1 &
    echo "[onstart] comfyui pid $!"
fi

for i in $(seq 1 300); do
    if ready; then
        echo "[onstart] ComfyUI готов через ${i}с"
        python3 /opt/ComfyUI/main.py --version >/dev/null 2>&1 || true
        exit 0
    fi
    sleep 1
done

echo "[onstart] ComfyUI не поднялся за 300с, хвост лога:"
tail -n 80 /workspace/comfyui.log 2>/dev/null
exit 1
