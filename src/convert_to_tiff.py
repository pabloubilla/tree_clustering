### Necessary for PLOS ONE submission

import os
from PIL import Image
import subprocess

# SETTINGS
INPUT_FOLDER = "output/figures_supporting/"
OUTPUT_DPI = 600
TIFF_COMPRESSION = "none"  # or "tiff_lzw" if you prefer light compression

def convert_png_to_tiff(file_path):
    """Convert PNG to TIFF using Pillow."""
    img = Image.open(file_path)
    tiff_path = os.path.splitext(file_path)[0] + ".tiff"
    img.save(tiff_path, dpi=(OUTPUT_DPI, OUTPUT_DPI), compression=TIFF_COMPRESSION)
    print(f"Converted: {file_path} → {tiff_path}")

def convert_pdf_to_tiff(file_path):
    """Convert PDF to TIFF using ImageMagick."""
    tiff_path = os.path.splitext(file_path)[0] + ".tiff"
    try:
        subprocess.run([
            "magick", "-density", str(OUTPUT_DPI),
            file_path, "-compress", TIFF_COMPRESSION, tiff_path
        ], check=True)
        print(f"Converted: {file_path} → {tiff_path}")
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")

def main():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".png", ".pdf")):
            full_path = os.path.join(INPUT_FOLDER, filename)
            if filename.lower().endswith(".png"):
                convert_png_to_tiff(full_path)
            elif filename.lower().endswith(".pdf"):
                convert_pdf_to_tiff(full_path)

if __name__ == "__main__":
    main()
