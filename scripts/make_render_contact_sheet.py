from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--thumb-width", type=int, default=180)
    args = parser.parse_args()

    render_paths = sorted(Path(args.render_dir).glob("*.png"))
    gt_paths = sorted(Path(args.gt_dir).glob("*.png"))
    count = min(args.count, len(render_paths), len(gt_paths))
    pad = 10
    label_h = 22
    pairs: list[Image.Image] = []

    for render_path, gt_path in zip(render_paths[:count], gt_paths[:count]):
        thumbs = []
        for path in (render_path, gt_path):
            image = Image.open(path).convert("RGB")
            height = round(image.height * args.thumb_width / image.width)
            thumbs.append(image.resize((args.thumb_width, height), Image.Resampling.LANCZOS))

        pair = Image.new(
            "RGB",
            (args.thumb_width * 2 + pad, max(image.height for image in thumbs) + label_h),
            (20, 20, 20),
        )
        draw = ImageDraw.Draw(pair)
        draw.text((2, 2), f"render {render_path.stem}", fill=(230, 230, 230))
        draw.text((args.thumb_width + pad + 2, 2), "gt", fill=(230, 230, 230))
        pair.paste(thumbs[0], (0, label_h))
        pair.paste(thumbs[1], (args.thumb_width + pad, label_h))
        pairs.append(pair)

    if not pairs:
        raise RuntimeError("No render/gt pairs found.")

    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    cell_w = max(pair.width for pair in pairs)
    cell_h = max(pair.height for pair in pairs)
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), (8, 10, 12))

    for index, pair in enumerate(pairs):
        x = pad + (index % cols) * (cell_w + pad)
        y = pad + (index // cols) * (cell_h + pad)
        sheet.paste(pair, (x, y))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(output_path)
    print(sheet.size)


if __name__ == "__main__":
    main()
