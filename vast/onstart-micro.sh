#!/bin/bash
# onstart-micro.sh: устанавливает torch + ComfyUI + скачивает чекпойнт.
# Образ содержит только CUDA + Python, всё остальное ставится здесь.
set -x
LOG=/workspace/onstart.log
mkdir -p /workspace
exec >>"$LOG" 2>&1

echo "[micro] start $(date -u)"
nvidia-smi || echo "[micro] nvidia-smi недоступен"

COMFY_DIR=/opt/ComfyUI
CKPT="$COMFY_DIR/models/checkpoints/waiNSFWIllustrious_v150.safetensors"
CKPT_SHA256="befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa"
CKPT_BYTES=6938040682

# === ЭТАП 1: torch + ComfyUI (если не установлены) ===
if ! python3 -c "import torch" 2>/dev/null; then
    echo "[micro] устанавливаю torch + torchvision + torchaudio (~5 ГБ)..."
    python3 -m pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
    echo "[micro] torch установлен: $(python3 -c 'import torch; print(torch.__version__)')"
fi

if [ ! -d "$COMFY_DIR/.git" ]; then
    echo "[micro] клонирую ComfyUI..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    python3 -m pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt"
    echo "[micro] ComfyUI установлен"
fi

# === ЭТАП 2: чекпойнт (если не скачан) ===
need_download=0
if [ ! -f "$CKPT" ]; then
    need_download=1
else
    actual_size=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    if [ "$actual_size" != "$CKPT_BYTES" ]; then
        need_download=1
    fi
fi

if [ "$need_download" = "1" ]; then
    echo "[micro] скачиваю чекпойнт (~6.9 ГБ)..."
    mkdir -p "$COMFY_DIR/models/checkpoints"
    ok=0
    for url in \
        "https://huggingface.co/LarryAIDraw/waiNSFWIllustrious_v150/resolve/main/waiNSFWIllustrious_v150.safetensors" \
        "https://huggingface.co/msagi/waiNSFWIllustrious_v150/resolve/main/waiNSFWIllustrious_v150.safetensors" \
    ; do
        echo "[micro] качаю: $url"
        start_dl=$(date +%s)
        if curl -fL --retry 3 --retry-delay 10 --connect-timeout 30 \
                -o "$CKPT" "$url?download=true"; then
            dl_time=$(( $(date +%s) - start_dl ))
            echo "[micro] скачано за ${dl_time}с"
            if echo "$CKPT_SHA256  $CKPT" | sha256sum -c -; then
                ok=1; break
            fi
            echo "[micro] sha256 не сошёлся"
        fi
        rm -f "$CKPT"
    done
    if [ "$ok" != "1" ]; then
        echo "[micro] СТОП: не удалось скачать чекпойнт"
        exit 1
    fi
fi

# === ЭТАП 3: запуск ComfyUI ===
ready() { curl -sf http://localhost:8188/system_stats >/dev/null 2>&1; }

if ready; then
    echo "[micro] ComfyUI уже отвечает"
else
    cd "$COMFY_DIR" || exit 1
    nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch > /workspace/comfyui.log 2>&1 &
    echo "[micro] comfyui pid $!"
fi

for i in $(seq 1 300); do
    if ready; then
        echo "[micro] ComfyUI готов через ${i}с"
        exit 0
    fi
    sleep 1
done

echo "[micro] ComfyUI не поднялся за 300с"
tail -n 80 /workspace/comfyui.log 2>/dev/null
exit 1
