"""Render a PDF to page PNGs for visual QA."""

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(args.pdf)
    for index in range(len(document)):
        page = document[index]
        page.render(scale=2.0).to_pil().save(output / f"page-{index + 1}.png")
    print(len(document))


if __name__ == "__main__":
    main()
