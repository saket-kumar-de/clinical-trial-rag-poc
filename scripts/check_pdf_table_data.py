import sys
sys.path.insert(0, ".")
import pymupdf as fitz

PDF_PATH = "data/raw/NCT03961204_protocol.pdf"
THIN_PAGES = [11, 12, 18, 19, 20, 27, 28, 82]

doc = fitz.open(PDF_PATH)
for page_num in THIN_PAGES:
    page = doc[page_num]
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    print(f"Page {page_num}: {len(images)} embedded image(s), {len(drawings)} vector drawing(s)")
    for img in images:
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        print(f"  image xref={xref}, size={pix.width}x{pix.height}, colorspace={pix.colorspace.name if pix.colorspace else 'none'}")
doc.close()