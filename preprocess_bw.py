import argparse
from pathlib import Path

import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def to_grayscale(image_path: Path, output_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[skip] Could not read: {image_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), gray)
    print(f"[ok] {image_path} -> {output_path}")
    return True


def process_single(input_path: Path, output_path: Path):
    if output_path.suffix == "":
        output_path = output_path / f"{input_path.stem}_bw{input_path.suffix}"
    return to_grayscale(input_path, output_path)


def process_directory(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            continue

        rel = path.relative_to(input_dir)
        out = output_dir / rel
        if to_grayscale(path, out):
            converted += 1
        else:
            skipped += 1

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert image(s) to black-and-white (grayscale)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input image file path or input directory path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output file or directory. "
            "If input is a file and output is a directory, filename gets '_bw' suffix."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    input_path = Path(args.input.strip().strip('"').strip("'")).expanduser().resolve()
    output_path = Path(args.output.strip().strip('"').strip("'")).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        process_single(input_path, output_path)
    else:
        process_directory(input_path, output_path)


if __name__ == "__main__":
    main()
