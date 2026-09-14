#!/bin/bash
# Onstart для comfyui-lite: качает чекпойнт с HuggingFace, потом запускает ComfyUI.
set -x
LOG=/workspace/onstart.log
mkdir -p /workspace
exec >>"$LOG" 2>&1

echo "[onstart-lite] start $(date -u)"
nvidia-smi || echo "[onstart-lite] nvidia-smi недоступен"

COMFY_DIR=/opt/ComfyUI
CKPT="$COMFY_DIR/models/checkpoints/waiNSFWIllustrious_v150.safetensors"
CKPT_SHA256="befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa"
CKPT_BYTES=6938040682

# --- Скачиваем чекпойнт если его нет или он битый
need_download=0
if [ ! -f "$CKPT" ]; then
    need_download=1
else
    actual_size=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    if [ "$actual_size" != "$CKPT_BYTES" ]; then
        echo "[onstart-lite] размер $actual_size != $CKPT_BYTES, перекачиваю"
        need_download=1
    else
        echo "[onstart-lite] чекпойнт на месте: $actual_size байт"
    fi
fi

if [ "$need_download" = "1" ]; then
    echo "[onstart-lite] скачиваю чекпойнт с HuggingFace (~6.9 ГБ)..."
    mkdir -p "$COMFY_DIR/models/checkpoints"
    ok=0
    for url in \
        "https://huggingface.co/LarryAIDraw/waiNSFWIllustrious_v150/resolve/main/waiNSFWIllustrious_v150.safetensors" \
        "https://huggingface.co/msagi/waiNSFWIllustrious_v150/resolve/main/waiNSFWIllustrious_v150.safetensors" \
    ; do
        echo "[onstart-lite] качаю: $url"
        start_dl=$(date +%s)
        if curl -fL --retry 3 --retry-delay 10 --connect-timeout 30 \
                -o "$CKPT" "$url?download=true"; then
            dl_time=$(( $(date +%s) - start_dl ))
            echo "[onstart-lite] скачано за ${dl_time}с"
            if echo "$CKPT_SHA256  $CKPT" | sha256sum -c -; then
                ok=1; break
            fi
            echo "[onstart-lite] sha256 не сошёлся, пробую следующее зеркало"
        fi
        rm -f "$CKPT"
    done
    if [ "$ok" != "1" ]; then
        echo "[onstart-lite] СТОП: не удалось скачать чекпойнт"
        exit 1
    fi
fi

# --- Запускаем ComfyUI
ready() { curl -sf http://localhost:8188/system_stats >/dev/null 2>&1; }

if ready; then
    echo "[onstart-lite] ComfyUI уже отвечает"
else
    echo "[onstart-lite] запускаю ComfyUI"
    cd "$COMFY_DIR" || { echo "[onstart-lite] нет $COMFY_DIR"; exit 1; }
    nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch > /workspace/comfyui.log 2>&1 &
    echo "[onstart-lite] comfyui pid $!"
fi

for i in $(seq 1 300); do
    if ready; then
        echo "[onstart-lite] ComfyUI готов через ${i}с"
        exit 0
    fi
    sleep 1
done

echo "[onstart-lite] ComfyUI не поднялся за 300с"
tail -n 80 /workspace/comfyui.log 2>/dev/null
exit 1
