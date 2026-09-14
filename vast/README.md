# vast/ — образ ComfyUI с запечённым каноном

Образ `ghcr.io/kirushanetypi/comfyui-canon:latest` — это ComfyUI, внутри
которого уже лежит чекпойнт канона (`waiNSFWIllustrious_v150.safetensors`).
Инстанс на Vast.ai не собирает окружение и не качает модель: поднялся —
сразу рисует.

## Что лежит здесь

| Файл | Зачем |
|------|-------|
| `Dockerfile.comfyui` | сборка образа (CUDA 12.8 runtime + python 3.11 + torch 2.11.0+cu128 + ComfyUI + чекпойнт) |
| `smoke_test.py` | пост-сборочная проверка: ComfyUI поднимается, чекпойнт виден, файл сходится по sha256 и читается как SDXL |
| `preset.json` | параметры инстанса Vast (образ, порт 8188, лимит цены/надёжности) |
| `onstart.sh` | onstart-скрипт Vast: ждёт ComfyUI, второй процесс не поднимает |

## Как собирается

Только в GitHub Actions — `.github/workflows/build-comfyui.yml` (push в `vast/**`
или ручной `workflow_dispatch`). Локально на VPS сборку не запускать: там занято
30 из 40 ГБ, а образ ~14 ГБ.

## Как проверить

```bash
# в CI это уже делает smoke-шаг; вручную на живом инстансе:
python3 vast/smoke_test.py --base-url http://<ip>:8188 --load
# анонимный pull без логина (пакет должен быть public):
TOKEN=$(curl -s 'https://ghcr.io/token?scope=repository:kirushanetypi/comfyui-canon:pull&service=ghcr.io' | jq -r .token)
curl -sI -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.oci.image.index.v1+json' \
  https://ghcr.io/v2/kirushanetypi/comfyui-canon/manifests/latest
```

## Что запечено и откуда

- Чекпойнт: single-file SDXL-версия Civitai-релиза **WAI NSFW Illustrious v15.0**,
  6 938 040 682 байта, sha256 `befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa`.
  Тот же файл с тем же sha256 лежит в нескольких независимых зеркалах на HF
  (`LarryAIDraw/waiNSFWIllustrious_v150`, `msagi/waiNSFWIllustrious_v150`);
  сборка берёт первое и падает, если хеш не сошёлся.
- Канон бота (config: `wai_illustrious_v15` → `John6666/wai-nsfw-illustrious-sdxl-v150-sdxl`)
  — это тот же релиз в diffusers-виде: у `John6666` и у `tonera/waiNSFWIllustrious_v150`
  совпадают sha256 unet/vae/text_encoder_2, то есть веса одни и те же.
