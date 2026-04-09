"""
Script de fine-tuning YOLOv8 para instrumentos cirúrgicos ginecológicos.
Otimizado para dataset pequeno (245 imagens) com RTX 5060 Ti (17GB VRAM).

Estratégias aplicadas:
  1. Modelo maior (yolov8s) — melhor capacidade sem custo proibitivo
  2. Freeze do backbone nas primeiras épocas — protege features pré-treinadas
  3. Augmentation agressivo — compensa dataset pequeno
  4. Warmup longo + Cosine LR — estabiliza treino com poucas amostras
  5. Cache de imagens na RAM — acelera cada época em ~3x

Uso:
    python models/yolov8/train/train.py
    python models/yolov8/train/train.py --model yolov8m --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

CONFIG = Path(__file__).parent.parent / "configs" / "data.yaml"
OUTPUT = Path(__file__).parent.parent / "weights"


def train(
    base_model: str = "yolov8s.pt",   # 's' tem ~3x mais params que 'n', ainda rápido com GPU
    epochs: int = 100,                 # Mais épocas compensam dataset pequeno
    imgsz: int = 640,
    batch: int = 32,                   # 17GB VRAM suporta batch alto
    freeze: int = 10,                  # Congelar as 10 primeiras camadas do backbone
):
    """
    Treina YOLOv8 com estratégias otimizadas para dataset pequeno.

    Args:
        base_model: Checkpoint de partida. 'yolov8s.pt' para melhor qualidade,
                    'yolov8n.pt' para prototipagem rápida.
        epochs    : Número de épocas. Com patience=15, para antes se não melhorar.
        imgsz     : Tamanho da imagem. 640 é padrão; 800 pode ajudar para objetos pequenos.
        batch     : Tamanho do batch. 32 é seguro para 17GB VRAM com imgsz=640.
        freeze    : Número de camadas do backbone a congelar nas primeiras épocas.
                    Evita destruir features COCO com dataset pequeno.
    """
    model = YOLO(base_model)

    results = model.train(
        data    = str(CONFIG),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = 0,            # GPU (RTX 5060 Ti)
        project = str(OUTPUT),
        name    = "surgical_tools_v1",

        # ── Convergência ──────────────────────────────────────────
        patience    = 15,       # Early stopping: para se mAP@50 não melhorar em 15 épocas
        optimizer   = "AdamW",  # Mais estável que SGD para datasets pequenos
        lr0         = 0.001,    # Learning rate inicial
        lrf         = 0.01,     # LR final = lr0 × lrf (cosine schedule)
        warmup_epochs = 5.0,    # Warmup longo — crítico com poucas amostras
        cos_lr      = True,     # Cosine annealing — evita oscilação no fim do treino

        # ── Transfer Learning ─────────────────────────────────────
        freeze      = freeze,   # Congela backbone; descongela automaticamente depois
        # (YOLOv8 só congela nas primeiras épocas quando freeze>0)

        # ── Augmentation agressivo (compensa dataset de 245 imagens) ─
        degrees     = 10.0,     # Rotação até 10° — instrumentos em qualquer ângulo
        translate   = 0.1,      # Translação 10%
        scale       = 0.5,      # Zoom in/out até 50%
        shear       = 2.0,      # Cisalhamento leve
        perspective = 0.0005,   # Distorção de perspectiva (simula câmera laparoscópica)
        flipud      = 0.3,      # Flip vertical — instrumentos aparecem em qualquer orientação
        fliplr      = 0.5,      # Flip horizontal
        mosaic      = 1.0,      # Mosaic ativo 100% — combina 4 imagens por batch
        mixup       = 0.1,      # MixUp leve — sobrepõe 2 imagens com peso
        copy_paste  = 0.1,      # Copy-paste: copia objetos entre imagens
        hsv_h       = 0.015,    # Variação de matiz (hue)
        hsv_s       = 0.7,      # Variação de saturação (imita iluminação cirúrgica variável)
        hsv_v       = 0.4,      # Variação de brilho

        # ── Performance ───────────────────────────────────────────
        cache       = True,     # Cache imagens na RAM (~245 × ~300KB ≈ 70MB — trivial)
        workers     = 4,        # Workers para DataLoader
        amp         = True,     # Automatic Mixed Precision (FP16) — 2x mais rápido na RTX

        # ── Saída ─────────────────────────────────────────────────
        save        = True,
        plots       = True,     # Gera gráficos de treino automaticamente
        verbose     = True,
        exist_ok    = True,
    )

    print(f"\nTreinamento concluído.")
    print(f"Melhor modelo : {results.save_dir}/weights/best.pt")
    print(f"mAP@50        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolov8s.pt", help="Modelo base YOLOv8")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch",  default=32, type=int)
    parser.add_argument("--imgsz",  default=640, type=int)
    parser.add_argument("--freeze", default=10, type=int)
    args = parser.parse_args()

    train(
        base_model = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        freeze     = args.freeze,
    )
