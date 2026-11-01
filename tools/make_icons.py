"""Render static/icons/favicon.svg to the PNG and ICO sizes (needs Playwright with Chrome and Pillow). Run from the project root."""
import io
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
d=Path('static/icons'); svg=(d/'favicon.svg').read_text()
with sync_playwright() as p:
    b=p.chromium.launch(channel='chrome'); pg=b.new_page(viewport={'width':512,'height':512})
    pg.set_content(f'<body style="margin:0;background:transparent">{svg.replace("<svg ","<svg width=512 height=512 ")}</body>')
    png=pg.screenshot(omit_background=True); b.close()
im=Image.open(io.BytesIO(png)).convert('RGBA')
im.resize((192,192),Image.LANCZOS).save(d/'icon-192.png'); im.save(d/'icon-512.png')
bg=Image.new('RGBA',(512,512),'#5b5bd6'); bg.alpha_composite(im.resize((420,420),Image.LANCZOS),(46,46))
bg.convert('RGB').resize((180,180),Image.LANCZOS).save(d/'apple-touch-icon.png')
im.save(d/'favicon.ico',sizes=[(16,16),(32,32),(48,48)])
