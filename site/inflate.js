/* "Inflate" any font into liquid chrome: draw the word, measure how deep each pixel sits inside the letters,
   and push the surface out by that depth (a balloon-shaped height map), then light it with a chrome studio. */
function inflateWord(canvas, { text = "parna", font = "Pacifico", size = 255, puff = 0.9, tint = 0xe2d2ff, stops = null } = {}) {
  if (!window.THREE) return null;
  const T = THREE, GW = 840, GH = 360;
  const c = document.createElement("canvas"); c.width = GW; c.height = GH; const g = c.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, GW, GH); g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
  let fs = size; g.font = `${fs}px "${font}"`; while (g.measureText(text).width > GW * .9 && fs > 20) { fs -= 4; g.font = `${fs}px "${font}"`; }
  g.fillText(text, GW / 2, GH / 2 + fs * .05);
  const px = g.getImageData(0, 0, GW, GH).data, inside = new Uint8Array(GW * GH);
  for (let i = 0; i < GW * GH; i++) inside[i] = px[i * 4] > 127;
  // chamfer distance transform (distance to the nearest outside pixel)
  const D = new Float32Array(GW * GH).map((_, i) => inside[i] ? 1e9 : 0), a = 1, b = 1.414;
  for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) { const i = y * GW + x; if (!D[i]) continue; let m = D[i];
    if (x) m = Math.min(m, D[i - 1] + a); if (y) { m = Math.min(m, D[i - GW] + a); if (x) m = Math.min(m, D[i - GW - 1] + b); if (x < GW - 1) m = Math.min(m, D[i - GW + 1] + b); } D[i] = m; }
  for (let y = GH - 1; y >= 0; y--) for (let x = GW - 1; x >= 0; x--) { const i = y * GW + x; if (!D[i]) continue; let m = D[i];
    if (x < GW - 1) m = Math.min(m, D[i + 1] + a); if (y < GH - 1) { m = Math.min(m, D[i + GW] + a); if (x < GW - 1) m = Math.min(m, D[i + GW + 1] + b); if (x) m = Math.min(m, D[i + GW - 1] + b); } D[i] = m; }
  let dmax = 1; for (const v of D) if (v > dmax) dmax = v;
  // balloon profile: round at the edges, soft on top; then a light blur so the surface is smooth
  let H = new Float32Array(GW * GH); for (let i = 0; i < D.length; i++) H[i] = D[i] ? Math.sqrt(Math.min(1, D[i] / dmax) * (2 - Math.min(1, D[i] / dmax))) : 0;
  // smooth the surface: separable box blur (3 passes) over the whole map, then keep only the inside
  const box = (A, r) => { const B = new Float32Array(A.length);
    for (let y = 0; y < GH; y++) { let acc = 0; for (let x = -r; x < GW + r; x++) { const xi = Math.min(GW - 1, Math.max(0, x)); acc += A[y * GW + xi];
      const xo = x - 2 * r - 1; if (xo >= -r) acc -= A[y * GW + Math.min(GW - 1, Math.max(0, xo))]; const xc = x - r; if (xc >= 0 && xc < GW) B[y * GW + xc] = acc / (2 * r + 1); } }
    const C = new Float32Array(A.length);
    for (let x = 0; x < GW; x++) { let acc = 0; for (let y = -r; y < GH + r; y++) { const yi = Math.min(GH - 1, Math.max(0, y)); acc += B[yi * GW + x];
      const yo = y - 2 * r - 1; if (yo >= -r) acc -= B[Math.min(GH - 1, Math.max(0, yo)) * GW + x]; const yc = y - r; if (yc >= 0 && yc < GH) C[yc * GW + x] = acc / (2 * r + 1); } }
    return C; };
  const rb = Math.max(2, Math.round(dmax / 3));
  for (let pass = 0; pass < 3; pass++) H = box(H, rb);
  let hm = 1e-6; for (let i = 0; i < H.length; i++) { if (!inside[i]) H[i] = 0; else if (H[i] > hm) hm = H[i]; }
  for (let i = 0; i < H.length; i++) if (H[i]) { const u = Math.min(1, H[i] / hm); H[i] = Math.pow(u, .75); }
  const sx = 6 / GW, pos = [], idx = [], map = new Int32Array(GW * GH).fill(-1);
  for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) { const i = y * GW + x;
    if (inside[i] || (x && inside[i - 1]) || (y && inside[i - GW])) { map[i] = pos.length / 3; pos.push((x - GW / 2) * sx, (GH / 2 - y) * sx, H[i] * dmax * sx * puff); } }
  for (let y = 0; y < GH - 1; y++) for (let x = 0; x < GW - 1; x++) { const i = y * GW + x, p = [map[i], map[i + 1], map[i + GW], map[i + GW + 1]];
    if (p.every((v) => v >= 0) && (inside[i] || inside[i + 1] || inside[i + GW] || inside[i + GW + 1])) idx.push(p[0], p[2], p[1], p[1], p[2], p[3]); }
  const geo = new T.BufferGeometry(); geo.setAttribute("position", new T.Float32BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
  const renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.outputEncoding = T.sRGBEncoding; renderer.toneMapping = T.ACESFilmicToneMapping;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(20, 1, .1, 100);
  const ec = document.createElement("canvas"); ec.width = 1024; ec.height = 512; const e = ec.getContext("2d");
  const v = e.createLinearGradient(0, 0, 0, 512);
  (stops || [[0, "#ffffff"], [.2, "#d9c2ff"], [.3, "#2a1648"], [.37, "#7d4fc0"], [.44, "#f2c4ec"], [.5, "#fdf6ff"], [.56, "#e6c9ff"], [.63, "#8a5ad0"], [.7, "#22123c"], [.8, "#c9a8f5"], [1, "#fff6fd"]]).forEach(([o, col]) => v.addColorStop(o, col));
  e.fillStyle = v; e.fillRect(0, 0, 1024, 512);
  const strip = (x, w, col) => { const h = e.createLinearGradient(x, 0, x + w, 0); h.addColorStop(0, "rgba(255,255,255,0)"); h.addColorStop(.5, col); h.addColorStop(1, "rgba(255,255,255,0)"); e.fillStyle = h; e.fillRect(x, 40, w, 430); };
  strip(120, 70, "rgba(255,255,255,.95)"); strip(330, 110, "rgba(255,150,210,.75)"); strip(560, 60, "rgba(255,255,255,.9)"); strip(700, 120, "rgba(140,170,255,.7)"); strip(900, 70, "rgba(255,210,240,.8)");
  const env = new T.CanvasTexture(ec); env.mapping = T.EquirectangularReflectionMapping; env.encoding = T.sRGBEncoding;
  scene.environment = new T.PMREMGenerator(renderer).fromEquirectangular(env).texture;
  const mesh = new T.Mesh(geo, new T.MeshPhysicalMaterial({ color: tint, metalness: 1, roughness: .07, clearcoat: 1, clearcoatRoughness: .04 }));
  const pivot = new T.Group(); pivot.add(mesh); scene.add(pivot);
  function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h;
    const tn = 2 * Math.tan(camera.fov * Math.PI / 360); camera.position.z = Math.max(6.3 / (tn * camera.aspect), 2.7 / tn); camera.updateProjectionMatrix(); }
  resize(); if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
  let mx = 0, my = 0; addEventListener("pointermove", (ev) => { mx = ev.clientX / innerWidth - .5; my = ev.clientY / innerHeight - .5; });
  let vis = true; new IntersectionObserver(([en]) => vis = en.isIntersecting).observe(canvas);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  (function frame(ms) { requestAnimationFrame(frame); if (!vis) return; const t = still ? 0 : ms / 1000;
    pivot.rotation.y = Math.sin(t * .5) * .18 + mx * .35; pivot.rotation.x = Math.sin(t * .37) * .07 + my * .2 - .05; renderer.render(scene, camera); })(0);
  return { resize };
}
