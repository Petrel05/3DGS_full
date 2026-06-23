from __future__ import annotations

import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3DGS WebGL Viewer</title>
<style>
html, body { margin:0; height:100%; overflow:hidden; background:#090b0f; color:#eef3f8; font-family:Arial, sans-serif; }
canvas { width:100vw; height:100vh; display:block; cursor:grab; touch-action:none; }
canvas:active { cursor:grabbing; }
#hud { position:fixed; left:14px; top:12px; padding:7px 9px; background:rgba(0,0,0,.42); border:1px solid rgba(255,255,255,.14); border-radius:6px; font-size:12px; letter-spacing:.02em; }
#tools { position:fixed; right:14px; top:12px; display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; max-width:460px; }
button { border:1px solid rgba(255,255,255,.18); background:rgba(12,16,22,.72); color:#eef3f8; border-radius:6px; padding:7px 9px; font-size:12px; cursor:pointer; }
button:hover { background:rgba(38,120,220,.78); }
#err { position:fixed; inset:0; display:none; place-items:center; padding:24px; color:#fff; background:#090b0f; font-size:15px; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="hud">3DGS WebGL · <span id="count"></span></div>
<div id="tools">
  <button data-view="front">front</button>
  <button data-view="side">side</button>
  <button data-view="top">top</button>
  <button id="sizeBtn">splat x1</button>
  <button id="depthBtn">depth color</button>
  <button id="thickBtn">thickness x3</button>
</div>
<div id="err"></div>
<script>
const canvas = document.getElementById('view');
const hudCount = document.getElementById('count');
const err = document.getElementById('err');

function showError(message) {
  err.style.display = 'grid';
  err.textContent = message;
}

let DATA;
try {
  DATA = __DATA__;
} catch (error) {
  showError('Could not parse embedded Gaussian data: ' + error.message);
  throw error;
}
hudCount.textContent = DATA.points.length.toLocaleString() + ' splats';

const gl = canvas.getContext('webgl2', {antialias: false, alpha: false, powerPreference: 'high-performance'}) ||
           canvas.getContext('webgl', {antialias: false, alpha: false, powerPreference: 'high-performance'});
if (!gl) {
  showError('WebGL unavailable');
  throw new Error('WebGL unavailable');
}

const vs = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute float aRadius;
attribute float aOpacity;
uniform float uYaw;
uniform float uPitch;
uniform float uZoom;
uniform float uAspect;
uniform float uPointScale;
uniform float uThickness;
uniform vec3 uCenter;
uniform float uSceneScale;
varying vec3 vColor;
varying float vOpacity;
varying float vDepth;
void main() {
  float cy = cos(uYaw), sy = sin(uYaw);
  float cp = cos(uPitch), sp = sin(uPitch);
  vec3 q = (aPosition - uCenter) * uSceneScale;
  q.y = -q.y;
  vec3 p = vec3(q.x, q.y, q.z * uThickness);
  float x1 = cy * p.x + sy * p.z;
  float z1 = -sy * p.x + cy * p.z;
  float y2 = cp * p.y - sp * z1;
  float z2 = sp * p.y + cp * z1;
  float depth = max(0.18, z2 + 3.0);
  vec2 ndc = vec2((x1 * uZoom) / (depth * uAspect), (y2 * uZoom) / depth);
  gl_Position = vec4(ndc, 1.0 - depth / 20.0, 1.0);
  gl_PointSize = clamp(aRadius * uPointScale * uZoom / depth, 1.0, 180.0);
  vColor = aColor;
  vOpacity = aOpacity;
  vDepth = clamp((z2 + 1.5) / 3.0, 0.0, 1.0);
}`;

const fs = `
precision mediump float;
varying vec3 vColor;
varying float vOpacity;
varying float vDepth;
uniform float uDepthMode;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float alpha = exp(-4.0 * r2) * vOpacity;
  vec3 depthColor = mix(vec3(0.1, 0.45, 1.0), vec3(1.0, 0.55, 0.1), vDepth);
  vec3 color = mix(vColor, depthColor, uDepthMode);
  gl_FragColor = vec4(color, alpha);
}`;

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    showError('Shader compile failed: ' + message);
    throw new Error(message);
  }
  return shader;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  const message = gl.getProgramInfoLog(program);
  showError('Shader link failed: ' + message);
  throw new Error(message);
}
gl.useProgram(program);

function flatten(items, width) {
  const out = new Float32Array(items.length * width);
  for (let i = 0; i < items.length; i++) out.set(items[i], i * width);
  return out;
}

const positions = flatten(DATA.points, 3);
const colors = flatten(DATA.colors, 3);
const radii = new Float32Array(DATA.radii);
const opacities = new Float32Array(DATA.opacities);
const basePointScale = Number.isFinite(DATA.viewer_point_scale) ? DATA.viewer_point_scale : 1.0;
let pointScaleFactor = 1.0;

function bounds(data) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.length; i += 3) {
    for (let j = 0; j < 3; j++) {
      const v = data[i + j];
      if (v < min[j]) min[j] = v;
      if (v > max[j]) max[j] = v;
    }
  }
  return {min, max};
}

const box = bounds(positions);
const center = [
  (box.min[0] + box.max[0]) * 0.5,
  (box.min[1] + box.max[1]) * 0.5,
  (box.min[2] + box.max[2]) * 0.5,
];
const extent = Math.max(
  box.max[0] - box.min[0],
  box.max[1] - box.min[1],
  box.max[2] - box.min[2],
  0.001
);
const sceneScale = 1.85 / extent;

function bindAttribute(name, data, size) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    showError('WebGL buffer upload failed for ' + name + ' (error ' + error + '). Try the 200k viewer or a desktop browser.');
    throw new Error('WebGL buffer upload failed: ' + error);
  }
  const location = gl.getAttribLocation(program, name);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
}

bindAttribute('aPosition', positions, 3);
bindAttribute('aColor', colors, 3);
bindAttribute('aRadius', radii, 1);
bindAttribute('aOpacity', opacities, 1);

const uniforms = {
  yaw: gl.getUniformLocation(program, 'uYaw'),
  pitch: gl.getUniformLocation(program, 'uPitch'),
  zoom: gl.getUniformLocation(program, 'uZoom'),
  aspect: gl.getUniformLocation(program, 'uAspect'),
  pointScale: gl.getUniformLocation(program, 'uPointScale'),
  thickness: gl.getUniformLocation(program, 'uThickness'),
  center: gl.getUniformLocation(program, 'uCenter'),
  sceneScale: gl.getUniformLocation(program, 'uSceneScale'),
  depthMode: gl.getUniformLocation(program, 'uDepthMode'),
};

gl.clearColor(0.035, 0.043, 0.06, 1.0);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

let yaw = 0.45;
let pitch = -0.12;
let zoom = 1.85;
let thickness = 1.0;
let depthMode = 0.35;
let dragging = false;
let lastX = 0;
let lastY = 0;
let dirty = true;

function resize() {
  const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const w = Math.floor(innerWidth * dpr);
  const h = Math.floor(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
  dirty = true;
}

function draw() {
  if (!dirty) return;
  dirty = false;
  resize();
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.uniform1f(uniforms.yaw, yaw);
  gl.uniform1f(uniforms.pitch, pitch);
  gl.uniform1f(uniforms.zoom, zoom);
  gl.uniform1f(uniforms.aspect, Math.max(0.001, canvas.width / canvas.height));
  gl.uniform1f(uniforms.pointScale, Math.max(canvas.width, canvas.height) * basePointScale * pointScaleFactor);
  gl.uniform1f(uniforms.thickness, thickness);
  gl.uniform3f(uniforms.center, center[0], center[1], center[2]);
  gl.uniform1f(uniforms.sceneScale, sceneScale);
  gl.uniform1f(uniforms.depthMode, depthMode);
  const chunk = 65536;
  for (let offset = 0; offset < DATA.points.length; offset += chunk) {
    gl.drawArrays(gl.POINTS, offset, Math.min(chunk, DATA.points.length - offset));
  }
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    showError('WebGL draw failed (error ' + error + '). Try the 200k viewer or a desktop browser.');
  }
}

function frame() {
  draw();
  requestAnimationFrame(frame);
}

canvas.addEventListener('pointerdown', (event) => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointerup', (event) => {
  dragging = false;
  try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
});
canvas.addEventListener('pointercancel', () => { dragging = false; });
canvas.addEventListener('pointermove', (event) => {
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.008;
  pitch = Math.max(-3.14159, Math.min(3.14159, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
  dirty = true;
});
canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(8.0, zoom));
  dirty = true;
}, {passive:false});
document.querySelectorAll('button[data-view]').forEach((button) => {
  button.addEventListener('click', () => {
    const view = button.dataset.view;
    if (view === 'front') { yaw = 0.0; pitch = 0.0; }
    if (view === 'side') { yaw = 1.5708; pitch = 0.0; }
    if (view === 'top') { yaw = 0.0; pitch = -1.5708; }
    dirty = true;
  });
});
document.getElementById('depthBtn').addEventListener('click', () => {
  depthMode = depthMode > 0.5 ? 0.0 : 1.0;
  dirty = true;
});
document.getElementById('sizeBtn').addEventListener('click', () => {
  const options = [0.45, 0.7, 1.0, 1.4];
  const idx = options.findIndex((value) => Math.abs(value - pointScaleFactor) < 1e-3);
  pointScaleFactor = options[(idx + 1) % options.length];
  document.getElementById('sizeBtn').textContent = 'splat x' + pointScaleFactor.toFixed(2).replace(/0$/, '').replace(/\\.$/, '');
  dirty = true;
});
document.getElementById('thickBtn').addEventListener('click', () => {
  thickness = thickness > 1.5 ? 1.0 : 3.0;
  dirty = true;
});
window.addEventListener('resize', resize);

resize();
frame();
</script>
</body>
</html>
"""


def write_viewer(path: str | Path, gaussians: dict) -> None:
    data = json.dumps(gaussians, ensure_ascii=False)
    Path(path).write_text(HTML_TEMPLATE.replace("__DATA__", data), encoding="utf-8")
