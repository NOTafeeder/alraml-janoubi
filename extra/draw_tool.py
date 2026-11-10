# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio", "numpy", "pillow", "pyproj"]
# ///
"""Builds draw_boundary.html: an offline page to draw the real boundary of al-Raml al-Janoubi on the
Sentinel-2 image, with named OpenStreetMap streets and facilities as guides. Output is GeoJSON (lon/lat)."""
import base64, io, json
from pathlib import Path
import numpy as np, rasterio
from PIL import Image
from pyproj import Transformer

R = Path(__file__).resolve().parent.parent
tif = sorted((R / "sentinel2").glob("*/s2_*.tif"))[-1]
with rasterio.open(tif) as s:
    T, W, H, crs = s.transform, s.width, s.height, s.crs
rgb_png = next(tif.parent.glob("rgb_*.png"))
buf = io.BytesIO(); Image.open(rgb_png).save(buf, "PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
def to_px(lon, lat):
    x, y = fwd.transform(lon, lat)
    c, r = ~T * (x, y)
    return round(c, 2), round(r, 2)

# affine pixel -> lon/lat fitted on a grid (error well under 1 m over 3.7 km)
cs, rs = np.meshgrid(np.linspace(0, W, 6), np.linspace(0, H, 6))
X, Y = T * (cs.ravel(), rs.ravel())
lon, lat = inv.transform(X, Y)
A = np.c_[cs.ravel(), rs.ravel(), np.ones(cs.size)]
cl = np.linalg.lstsq(A, lon, rcond=None)[0].tolist()
ct = np.linalg.lstsq(A, lat, rcond=None)[0].tolist()
err = max(np.abs(A @ cl - lon).max() * 90000, np.abs(A @ ct - lat).max() * 111000)
print("affine max error m", round(err, 2))

osm = json.loads((R / "data_extra" / "osm.json").read_text())
import sys; sys.path.insert(0, str(R / "extra"))
from page_data import clean_name
roads, pois = [], []
for e in osm["elements"]:
    t = e.get("tags", {}); g = e.get("geometry")
    if "highway" in t and g:
        roads.append({"n": clean_name(t), "p": [to_px(p["lon"], p["lat"]) for p in g]})
    k = t.get("amenity") or t.get("leisure")
    if k and clean_name(t) and "lat" in e:
        pois.append({"n": clean_name(t), "p": to_px(e["lon"], e["lat"])})
    elif k and clean_name(t) and g:
        pois.append({"n": clean_name(t), "p": to_px(np.mean([p["lon"] for p in g]), np.mean([p["lat"] for p in g]))})

draft = []
dp = R / "data_extra" / "boundary_draft.geojson"
if dp.exists():
    ring = json.loads(dp.read_text(encoding="utf-8"))["geometry"]["coordinates"][0][:-1]
    draft = [list(to_px(lo, la)) for lo, la in ring]
DATA = json.dumps({"img": img, "w": W, "h": H, "cl": cl, "ct": ct, "roads": roads, "pois": pois, "draft": draft}, ensure_ascii=False)
HTML = r"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>رسم حدود الحي</title>
<style>
:root{--bg:#f4f2ec;--card:#fff;--ink:#1c2126;--muted:#5b636c;--line:#e4dfd4;--accent:#1d6a86;--chip:#eef3f5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121518;--card:#1b1f24;--ink:#e7e9ec;--muted:#a2a9b1;--line:#2d333a;--accent:#6cb6d4;--chip:#232b33}}
:root[data-theme="dark"]{--bg:#121518;--card:#1b1f24;--ink:#e7e9ec;--muted:#a2a9b1;--line:#2d333a;--accent:#6cb6d4;--chip:#232b33}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI",Tahoma,Arial,sans-serif;line-height:1.7}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:4px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
button{font:inherit;border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:999px;padding:5px 14px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
label{font-size:.9rem;color:var(--muted)}
#wrap{position:relative;width:100%;aspect-ratio:__W__/__H__;border-radius:10px;overflow:visible;background:#000;cursor:crosshair;touch-action:none}
#wrap img{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
#wrap svg{position:absolute;inset:0;width:100%;height:100%;direction:ltr;overflow:visible}
.road{fill:none;stroke:#fff;stroke-opacity:.55;stroke-width:1.2;vector-effect:non-scaling-stroke}
.road.named{stroke:#ffd166;stroke-opacity:.85;stroke-width:2}
.road.named:hover{stroke:#ff4d6d;stroke-width:4}
.poi{fill:#fff;stroke:#000;stroke-width:.6}
.poly{fill:rgba(255,77,109,.18);stroke:#ff4d6d;stroke-width:2.5;vector-effect:non-scaling-stroke}
.vtx{fill:#ff4d6d;stroke:#fff;stroke-width:1.5;vector-effect:non-scaling-stroke;cursor:move}
#tip{position:absolute;background:#000c;color:#fff;font-size:.85rem;padding:2px 8px;border-radius:6px;pointer-events:none;transform:translate(-50%,-140%);white-space:nowrap}
textarea{width:100%;min-height:90px;font-family:Consolas,monospace;font-size:.8rem;direction:ltr;background:var(--chip);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}
.muted{color:var(--muted);font-size:.9rem}
ol{margin:4px 0;padding-inline-start:22px}
</style></head><body><main>
<h1>ارسم حدود الرمل الجنوبي</h1>
<div class="card">
<ol>
<li>المسودة المرسومة باللون الأحمر مبنية على وصفك. اسحب أي نقطة لتصحيحها، أو اضغط «مسح الكل» وارسم من جديد بالضغط على الصورة.</li>
<li>اضغط على الصورة لتضيف نقطة في آخر الحدود. الشوارع المسمّاة باللون الأصفر؛ مرّر المؤشر فوقها لترى اسمها.</li>
<li>تحتاج 3 نقاط على الأقل. اسحب أي نقطة لتحريكها. «تراجع» يحذف آخر نقطة.</li>
<li>عند الانتهاء اضغط «نسخ النتيجة» والصقها في المحادثة، أو «حفظ ملف».</li>
</ol>
</div>
<div class="card">
<div class="bar">
<button id="undo">تراجع</button><button id="clear">مسح الكل</button><button id="reset">إعادة المسودة</button>
<label><input type="checkbox" id="showRoads" checked> الشوارع</label>
<label><input type="checkbox" id="showPois" checked> المرافق</label>
<span class="muted" id="info"></span>
</div>
<div id="wrap"><img id="img" alt="صورة القمر الصناعي 2026"><svg id="svg" viewBox="0 0 __W__ __H__" preserveAspectRatio="none"></svg><div id="tip" hidden></div></div>
</div>
<div class="card">
<div class="bar"><button class="primary" id="copy">نسخ النتيجة</button><button id="save">حفظ ملف</button><span class="muted" id="msg"></span></div>
<textarea id="out" readonly placeholder="ستظهر هنا حدود الحي بصيغة GeoJSON"></textarea>
<p class="muted">الصورة: Sentinel-2، آب 2026. الشوارع والمرافق: © مساهمو OpenStreetMap. الإحداثيات بالدرجات (WGS84).</p>
</div>
</main>
<script>
const D = __DATA__;
const svg = document.getElementById("svg"), wrap = document.getElementById("wrap"), tip = document.getElementById("tip");
document.getElementById("img").src = D.img;
const NS = "http://www.w3.org/2000/svg";
const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };
const gRoads = el("g", {}), gPois = el("g", {}), gPoly = el("g", {});
svg.append(gRoads, gPois, gPoly);
D.roads.forEach(r => { const p = el("polyline", {points: r.p.map(q => q.join(",")).join(" "), class: "road" + (r.n ? " named" : "")});
  if (r.n) { p.dataset.n = r.n; } gRoads.append(p); });
D.pois.forEach(q => { const c = el("circle", {cx: q.p[0], cy: q.p[1], r: 2.2, class: "poi"}); c.dataset.n = q.n; gPois.append(c); });
const toLL = (c, r) => [D.cl[0] * c + D.cl[1] * r + D.cl[2], D.ct[0] * c + D.ct[1] * r + D.ct[2]];
let pts = (D.draft || []).map(p => p.slice()), drag = -1;
const evPx = e => { const b = svg.getBoundingClientRect(); return [(e.clientX - b.left) / b.width * D.w, (e.clientY - b.top) / b.height * D.h]; };
function draw(){
  gPoly.innerHTML = "";
  if (pts.length > 1) gPoly.append(el(pts.length > 2 ? "polygon" : "polyline", {points: pts.map(p => p.join(",")).join(" "), class: "poly"}));
  pts.forEach((p, i) => { const c = el("circle", {cx: p[0], cy: p[1], r: 3.2, class: "vtx"}); c.dataset.i = i; gPoly.append(c); });
  let area = 0;
  if (pts.length > 2) { for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; area += a[0] * b[1] - b[0] * a[1]; } area = Math.abs(area) / 2 * 100 / 1e6; }
  document.getElementById("info").textContent = `${pts.length} نقطة` + (pts.length > 2 ? ` — المساحة ≈ ${area.toFixed(2)} كم²` : "");
  const out = document.getElementById("out");
  if (pts.length > 2) {
    const ring = pts.map(p => toLL(...p).map(v => +v.toFixed(6))); ring.push(ring[0]);
    out.value = JSON.stringify({type: "Feature", properties: {name: "الرمل الجنوبي", source: "رسم يدوي على Sentinel-2"}, geometry: {type: "Polygon", coordinates: [ring]}});
  } else out.value = "";
}
svg.addEventListener("pointerdown", e => {
  if (e.target.classList.contains("vtx")) { drag = +e.target.dataset.i; svg.setPointerCapture(e.pointerId); return; }
  pts.push(evPx(e).map(v => +v.toFixed(1))); draw();
});
svg.addEventListener("pointermove", e => {
  if (drag >= 0) { pts[drag] = evPx(e).map(v => +v.toFixed(1)); draw(); return; }
  const n = e.target.dataset && e.target.dataset.n;
  if (n) { const b = wrap.getBoundingClientRect(); tip.textContent = n; tip.hidden = false; tip.style.left = (e.clientX - b.left) + "px"; tip.style.top = (e.clientY - b.top) + "px"; }
  else tip.hidden = true;
});
svg.addEventListener("pointerup", () => drag = -1);
document.getElementById("undo").onclick = () => { pts.pop(); draw(); };
document.getElementById("clear").onclick = () => { pts = []; draw(); };
document.getElementById("reset").onclick = () => { pts = (D.draft || []).map(p => p.slice()); draw(); };
document.getElementById("showRoads").onchange = e => gRoads.style.display = e.target.checked ? "" : "none";
document.getElementById("showPois").onchange = e => gPois.style.display = e.target.checked ? "" : "none";
document.getElementById("copy").onclick = async () => {
  const t = document.getElementById("out").value; const m = document.getElementById("msg");
  if (!t) { m.textContent = "ارسم 3 نقاط على الأقل أولاً."; return; }
  try { await navigator.clipboard.writeText(t); m.textContent = "نُسخت. الصقها في المحادثة."; }
  catch (e) { const o = document.getElementById("out"); o.select(); document.execCommand("copy"); m.textContent = "نُسخت. الصقها في المحادثة."; }
};
document.getElementById("save").onclick = () => {
  const t = document.getElementById("out").value; if (!t) return;
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([t], {type: "application/geo+json"}));
  a.download = "alraml_boundary.geojson"; a.click();
};
draw();
</script></body></html>"""
html = HTML.replace("__DATA__", DATA).replace("__W__", str(W)).replace("__H__", str(H))
(R / "draw_boundary.html").write_text(html, encoding="utf-8")
print("wrote draw_boundary.html", round(len(html) / 1e6, 2), "MB", len(roads), "roads", len(pois), "pois")
