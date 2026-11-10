"""Embedded IBM Plex Sans Arabic (SIL Open Font License, fonts/OFL.txt) as @font-face rules with data: URLs,
so the pages keep working offline. Used by build_page.py and extra/draw_tool.py."""
import base64
from pathlib import Path

FONTS = Path(__file__).resolve().parent.parent / "fonts"
RANGES = {
    "arabic": "U+0600-06FF, U+0750-077F, U+0870-088E, U+0890-0891, U+0898-08E1, U+08E3-08FF, U+200C-200E, "
              "U+2010-2011, U+204F, U+2E41, U+FB50-FDFF, U+FE70-FE74, U+FE76-FEFC",
    "latin": "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, "
             "U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD",
}


def font_css():
    rules = []
    for sub, rng in RANGES.items():
        for w in (400, 600):
            f = FONTS / f"ibm-plex-sans-arabic-{sub}-{w}-normal.woff2"
            if not f.exists():
                continue
            b64 = base64.b64encode(f.read_bytes()).decode()
            rules.append(f"@font-face{{font-family:'Plex Arabic';font-style:normal;font-weight:{w};font-display:swap;"
                         f"src:url(data:font/woff2;base64,{b64}) format('woff2');unicode-range:{rng}}}")
    return "\n".join(rules)
