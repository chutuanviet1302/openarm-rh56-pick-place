const canvas = document.querySelector('#table');
const ctx = canvas.getContext('2d');
const form = document.querySelector('#controls');
const statusBox = document.querySelector('#status');
const resultLink = document.querySelector('#result');
const button = form.querySelector('button[type="submit"]');
const bounds = { x: [-0.25, 0.75], y: [-0.55, 0.55] };
const base = [-0.03, 0];
const baseHalf = [0.13, 0.10];
// Reach polygons are IK-verified by scripts/sweep_reach_map.py (same wrist targets,
// yaws, seeds and joint-margin gate the real planner uses) and loaded from
// reach_map.json. VALIDATION snaps to the nearest grid cell's boolean -- no hull
// optimism; the convex hull below is only drawn as a visual guide.
let reachAreas = null;   // {right: {grasp: hull, place: hull}, left: ...} for drawing
let reachMap = null;     // raw grid for validation
const pad = 54;
const fields = {
  object: [document.querySelector('#object-x'), document.querySelector('#object-y')],
  basket: [document.querySelector('#basket-x'), document.querySelector('#basket-y')],
};
let selected = 'object';

function value(name) { return fields[name].map(input => Number(input.value)); }
function point([x, y]) {
  return [pad + (x - bounds.x[0]) / (bounds.x[1] - bounds.x[0]) * (canvas.width - 2 * pad),
    canvas.height - pad - (y - bounds.y[0]) / (bounds.y[1] - bounds.y[0]) * (canvas.height - 2 * pad)];
}
function world(px, py) {
  return [bounds.x[0] + (px - pad) / (canvas.width - 2 * pad) * (bounds.x[1] - bounds.x[0]),
    bounds.y[0] + (canvas.height - pad - py) / (canvas.height - 2 * pad) * (bounds.y[1] - bounds.y[0])];
}
function polygonPath(points) {
  ctx.beginPath(); points.forEach((p, i) => { const [x,y] = point(p); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); }); ctx.closePath();
}
function inPolygon(position, polygon) {
  let inside = false;
  for (let i=0,j=polygon.length-1; i<polygon.length; j=i++) {
    const [xi,yi]=polygon[i], [xj,yj]=polygon[j], [x,y]=position;
    if ((yi>y)!==(yj>y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi) inside=!inside;
  }
  return inside;
}
function convexHull(points) {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const half = (list) => {
    const out = [];
    for (const p of list) { while (out.length >= 2 && cross(out[out.length-2], out[out.length-1], p) <= 0) out.pop(); out.push(p); }
    return out;
  };
  return half(pts).slice(0, -1).concat(half(pts.reverse()).slice(0, -1));
}
function canReach(side, kind, position) {
  if (!reachMap) return false;
  const [x, y] = position;
  const [sx, sy] = reachMap.step;
  if (x < reachMap.xs[0] || x > reachMap.xs[reachMap.xs.length-1] || y < reachMap.ys[0] || y > reachMap.ys[reachMap.ys.length-1]) return false;
  const ix = Math.round((x - reachMap.xs[0]) / sx), iy = Math.round((y - reachMap.ys[0]) / sy);
  return !!reachMap[side][kind][ix][iy];
}
fetch('reach_map.json', { cache: 'no-store' }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }).then(map => {
  reachMap = map;
  reachAreas = {};
  for (const side of ['right', 'left']) {
    reachAreas[side] = {};
    for (const kind of ['grasp', 'place']) {
      const cells = [];
      map.xs.forEach((x, ix) => map.ys.forEach((y, iy) => { if (map[side][kind][ix][iy]) cells.push([x, y]); }));
      reachAreas[side][kind] = convexHull(cells);
    }
  }
  draw();
}).catch(() => draw());
function drawMarker(name, color, label) {
  const [x, y] = point(value(name));
  ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  if (name === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.stroke(); }
  ctx.fillStyle = '#081014'; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center'; ctx.fillText(label, x, y + 5);
}
function validateLayout() {
  const object = value('object'), basket = value('basket');
  const inside = point => point.every((v, i) => Number.isFinite(v) && v >= bounds[i ? 'y' : 'x'][0] && v <= bounds[i ? 'y' : 'x'][1]);
  const overlapsBase = (point, half) => point.every((v, i) => Math.abs(v-base[i]) < baseHalf[i]+half);
  const distance = Math.hypot(object[0]-basket[0], object[1]-basket[1]);
  let message = 'BỐ TRÍ HỢP LỆ';
  if (!reachMap) message = 'ĐANG TẢI BẢN ĐỒ VÙNG IK…';
  else if (!inside(object) || !inside(basket)) message = 'ĐIỂM NẰM NGOÀI MẶT BÀN';
  else if (overlapsBase(object, .0354) || overlapsBase(basket, .10)) message = 'VẬT HOẶC RỔ CHỒNG LÊN ĐẾ';
  else if (distance < .15) message = 'VẬT VÀ RỔ PHẢI CÁCH NHAU 15 CM';
  else if (!['right','left'].some(side => canReach(side, 'grasp', object))) message = 'PICK NGOÀI VÙNG IK ĐÃ KIỂM CHỨNG';
  else if (!['right','left'].some(side => canReach(side, 'place', basket))) message = 'PLACE NGOÀI VÙNG IK ĐÃ KIỂM CHỨNG';
  document.querySelector('#object-distance').textContent = `${Math.hypot(object[0]-base[0], object[1]-base[1]).toFixed(2)} m`;
  document.querySelector('#basket-distance').textContent = `${Math.hypot(basket[0]-base[0], basket[1]-base[1]).toFixed(2)} m`;
  document.querySelector('#pair-distance').textContent = `${distance.toFixed(2)} m`;
  const state = document.querySelector('#layout-state'); state.textContent = message; state.className = message === 'BỐ TRÍ HỢP LỆ' ? 'valid' : 'invalid';
  button.disabled = state.className === 'invalid';
  return state.className === 'valid';
}
function draw() {
  ctx.fillStyle = '#351d24'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (reachAreas) {
    polygonPath(reachAreas.right[selected]); ctx.fillStyle = '#123b34'; ctx.fill(); ctx.strokeStyle='#62e8bd'; ctx.lineWidth=4; ctx.stroke();
    polygonPath(reachAreas.left[selected]); ctx.fillStyle = '#203551'; ctx.fill(); ctx.strokeStyle='#6eaeff'; ctx.lineWidth=4; ctx.stroke();
  }
  ctx.strokeStyle = '#5e343b'; ctx.lineWidth = 1;
  for (let x = bounds.x[0]; x <= bounds.x[1] + .001; x += .1) { const [px] = point([x, 0]); ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, canvas.height-pad); ctx.stroke(); }
  for (let y = bounds.y[0]; y <= bounds.y[1] + .001; y += .1) { const [,py] = point([0, y]); ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(canvas.width-pad, py); ctx.stroke(); }
  const [left, top] = point([-0.16, 0.10]); const [right, bottom] = point([0.10, -0.10]);
  ctx.fillStyle = '#111b21'; ctx.fillRect(left, top, right-left, bottom-top);
  ctx.strokeStyle = '#ff7668'; ctx.lineWidth = 3; ctx.setLineDash([8,6]); ctx.strokeRect(left, top, right-left, bottom-top); ctx.setLineDash([]);
  ctx.fillStyle = '#eff4f7'; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'center'; ctx.fillText('ROBOT', (left+right)/2, (top+bottom)/2+4);
  drawMarker('object', '#ffbe4d', 'A'); drawMarker('basket', '#50df8b', 'B');
  validateLayout();
}
function select(name) {
  selected = name;
  document.querySelector('#select-object').classList.toggle('active', name === 'object');
  document.querySelector('#select-basket').classList.toggle('active', name === 'basket');
  document.querySelector('#map-mode').textContent = name === 'object' ? 'PICK MAP' : 'PLACE MAP';
  draw();
}
function moveSelected(event) {
  const rect = canvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * canvas.width / rect.width;
  const py = (event.clientY - rect.top) * canvas.height / rect.height;
  const [x, y] = world(px, py);
  fields[selected][0].value = Math.max(bounds.x[0], Math.min(bounds.x[1], x)).toFixed(2);
  fields[selected][1].value = Math.max(bounds.y[0], Math.min(bounds.y[1], y)).toFixed(2);
  draw();
}
canvas.addEventListener('pointerdown', event => {
  const rect = canvas.getBoundingClientRect();
  const cursor = [(event.clientX-rect.left)*canvas.width/rect.width, (event.clientY-rect.top)*canvas.height/rect.height];
  const nearest = ['object','basket'].sort((a,b) => Math.hypot(...point(value(a)).map((v,i)=>v-cursor[i])) - Math.hypot(...point(value(b)).map((v,i)=>v-cursor[i])))[0];
  if (Math.hypot(...point(value(nearest)).map((v,i)=>v-cursor[i])) < 32) select(nearest);
  canvas.setPointerCapture(event.pointerId); moveSelected(event);
});
canvas.addEventListener('pointermove', event => { if (canvas.hasPointerCapture(event.pointerId)) moveSelected(event); });
document.querySelector('#select-object').addEventListener('click', () => select('object'));
document.querySelector('#select-basket').addEventListener('click', () => select('basket'));
Object.values(fields).flat().forEach(input => input.addEventListener('input', draw));

async function poll(job) {
  const response = await fetch(`/api/trajectory/${job}`); const state = await response.json();
  statusBox.textContent = state.message;
  if (state.status === 'queued' || state.status === 'running') { setTimeout(() => poll(job), 1200); return; }
  button.disabled = false;
  if (state.status === 'complete') {
    statusBox.textContent = `${state.passed ? '✓ Thành công' : '✕ Chưa đạt'} — ${state.message}`;
    resultLink.href = `./#${state.episode}`; resultLink.hidden = false;
  }
}
form.addEventListener('submit', async event => {
  event.preventDefault(); if (!validateLayout()) return; button.disabled = true; resultLink.hidden = true; statusBox.textContent = 'Đang gửi…';
  try {
    const response = await fetch('/api/trajectory', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({object:value('object'), basket:value('basket')}) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error); statusBox.textContent = `Route: ${data.route}`; poll(data.job);
  } catch (error) { statusBox.textContent = `Lỗi: ${error.message}`; button.disabled = false; }
});
draw();
