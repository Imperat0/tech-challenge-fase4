"""
Conversão: m2caiSeg (máscaras PNG segmentação) → Labels YOLO (.txt bounding boxes)

O dataset m2caiSeg contém:
  train/images/*.jpg       — frames de cirurgia laparoscópica
  train/groundtruth/*_gt.png — máscaras de segmentação (pixel branco = instrumento)

O YOLOv8 espera labels no formato:
  <class_id> <x_center> <y_center> <width> <height>   (valores normalizados 0-1)

Como m2caiSeg tem segmentação (não classes múltiplas), todos os instrumentos
recebem class_id = 0 ("surgical_instrument").

Uso:
    python convert_m2cai_to_yolo.py

Saída gerada em:
    m2caiSeg/train/labels/*.txt
    m2caiSeg/test/labels/*.txt
"""

import cv2
import numpy as np
from pathlib import Path


DATASET_ROOT = Path(__file__).parent / "m2caiSeg"
CLASS_ID = 0  # classe única: "surgical_instrument"


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    Lê imagem de caminho com caracteres especiais (workaround OpenCV/Windows).
    Usa np.frombuffer para evitar limitação do cv2.imread com Unicode.
    """
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def mask_to_yolo_bbox(mask_path: Path, image_path: Path) -> list[str]:
    """
    Lê uma máscara PNG binária e converte cada componente conectado
    em um bounding box no formato YOLO.

    Returns:
        Lista de strings no formato "class_id x_c y_c w h"
    """
    mask = _imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Threshold: pixels > 10 = instrumento (branco)
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Usa dimensões da imagem original para normalização
    img = _imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        h, w = binary.shape
    else:
        h, w = img.shape[:2]

    # Encontrar componentes conectados (cada instrumento separado)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    lines = []
    for i in range(1, num_labels):  # i=0 é background
        x, y, bw, bh, area = stats[i]

        # Ignorar ruídos muito pequenos (< 0.1% da imagem)
        if area < (h * w * 0.001):
            continue

        # Normalizar para 0-1
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return lines


def convert_split(split: str) -> tuple[int, int]:
    """Converte um split (train/test) completo. Retorna (convertidos, pulados)."""
    gt_dir = DATASET_ROOT / split / "groundtruth"
    img_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"
    label_dir.mkdir(exist_ok=True)

    if not gt_dir.exists():
        print(f"  [AVISO] Pasta não encontrada: {gt_dir}")
        return 0, 0

    converted = 0
    skipped = 0

    for mask_path in sorted(gt_dir.glob("*_gt.png")):
        # Encontrar imagem correspondente
        stem = mask_path.stem.replace("_gt", "")
        img_path = img_dir / f"{stem}.jpg"

        if not img_path.exists():
            skipped += 1
            continue

        lines = mask_to_yolo_bbox(mask_path, img_path)

        # Salvar label (mesmo sem detecções, arquivo vazio é válido para YOLO)
        label_path = label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines))
        converted += 1

    return converted, skipped


def main():
    print("=" * 55)
    print("  Conversão m2caiSeg → Labels YOLO")
    print("=" * 55)

    for split in ["train", "test", "trainval"]:
        print(f"\n[{split.upper()}]")
        conv, skip = convert_split(split)
        print(f"  Convertidos : {conv}")
        print(f"  Pulados     : {skip}")

    print("\nConversão concluída!")
    print(f"Labels salvas em: {DATASET_ROOT}/*/labels/")

    # Gerar data.yaml para YOLOv8
    yaml_content = f"""# Dataset m2caiSeg convertido para YOLOv8
path: {DATASET_ROOT.resolve()}
train: train/images
val: test/images

nc: 1
names:
  0: surgical_instrument
"""
    yaml_path = DATASET_ROOT / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"data.yaml gerado em: {yaml_path}")


if __name__ == "__main__":
    main()
