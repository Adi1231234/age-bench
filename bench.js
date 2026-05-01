import { pipeline, env, RawImage } from 'https://esm.sh/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const $ = (id) => document.getElementById(id);
const log = (msg) => { const el = $('log'); el.textContent += msg + '\n'; el.scrollTop = el.scrollHeight; };
const setProgress = (done, total) => {
  $('progBar').style.width = (total ? (done / total * 100) : 0) + '%';
  $('progText').textContent = `${done}/${total}`;
};
const setVariantProgress = (done, total) => {
  $('progBarVar').style.width = (total ? (done / total * 100) : 0) + '%';
  $('progTextVar').textContent = `${done}/${total}`;
};

const variantState = {};
function setVariantState(id, state) { variantState[id] = state; renderStrip(); refreshVariantProgress(); }
function refreshVariantProgress() {
  const done = Object.values(variantState).filter(s => s === 'done' || s === 'cached' || s === 'fail' || s === 'timeout').length;
  setVariantProgress(done, VARIANTS.length);
}
function renderStrip() {
  const strip = $('strip');
  strip.innerHTML = '';
  VARIANTS.forEach((v) => {
    const state = variantState[v.id] || 'queued';
    const chip = document.createElement('span');
    chip.className = 'chip ' + state;
    const icon = { done: '✓', running: '▶', cached: '📦', fail: '✗', timeout: '⏱', queued: '⏳' }[state];
    chip.innerHTML = `<span class="icon">${icon}</span><span class="name">${v.id}</span>`;
    strip.appendChild(chip);
  });
}

function setBanner(state, badgeText, title, meta, right) {
  const ban = $('currentBanner');
  ban.className = 'current-banner' + (state === 'idle' ? ' idle' : '');
  $('banBadge').textContent = badgeText;
  $('banTitle').textContent = title;
  $('banMeta').textContent = meta || '';
  $('banRight').innerHTML = right || '';
}

const FAIRFACE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70'];

function strictKidFromBuckets(label) {
  return label === '0-2' || label === '3-9';
}
function lenientKidFromBuckets(label) {
  return label === '0-2' || label === '3-9' || label === '10-19';
}

function fairfaceProbWeighted(out, weights, threshold) {
  // weights = [w_0-2, w_3-9, w_10-19] applied to kid prob mass; threshold to declare "kid"
  const map = Object.fromEntries(out.map(r => [r.label, r.score]));
  const kid = (map['0-2'] || 0) * weights[0] + (map['3-9'] || 0) * weights[1] + (map['10-19'] || 0) * weights[2];
  return kid > threshold;
}
function fairfaceMargin(out, marginRequired) {
  // declare kid if max(kid bucket prob) - max(adult bucket prob) > marginRequired
  const map = Object.fromEntries(out.map(r => [r.label, r.score]));
  const kidMax = Math.max(map['0-2'] || 0, map['3-9'] || 0);
  const adultMax = Math.max(map['20-29'] || 0, map['30-39'] || 0, map['40-49'] || 0, map['50-59'] || 0, map['60-69'] || 0, map['more than 70'] || 0);
  return kidMax - adultMax > marginRequired;
}

function decideYoloFaceAge(out, mode) {
  if (!Array.isArray(out) || out.length === 0) return false;
  const isKidLabel = (label) => {
    const l = String(label || '').toLowerCase();
    return l.includes('0-14') || l.startsWith('0') || l === 'kid' || l === 'child';
  };
  const isMidLabel = (label) => {
    const l = String(label || '').toLowerCase();
    return l.includes('15-22') || l.includes('teen');
  };
  if (mode === 'strict') {
    const best = out.slice().sort((a, b) => boxArea(b) - boxArea(a))[0];
    return best ? isKidLabel(best.label) : false;
  }
  if (mode === 'lenient') {
    const best = out.slice().sort((a, b) => boxArea(b) - boxArea(a))[0];
    return best ? (isKidLabel(best.label) || isMidLabel(best.label)) : false;
  }
  if (mode === 'vote') {
    let kidVotes = 0, adultVotes = 0;
    for (const det of out) {
      if (isKidLabel(det.label)) kidVotes += det.score || 1;
      else if (!isMidLabel(det.label)) adultVotes += det.score || 1;
    }
    return kidVotes > adultVotes;
  }
  return false;
}
function boxArea(d) { return (d.box?.xmax - d.box?.xmin) * (d.box?.ymax - d.box?.ymin) || 0; }

const VARIANTS = [
  // ==================== FairFace - 7 weighted variants (q8 wasm + fp16 webgpu) ====================
  { id: 'fairface-q8-strict',          engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'wasm',   decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q8-lenient',         engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'wasm',   decide: (out) => lenientKidFromBuckets(out[0].label) },
  { id: 'fairface-q8-prob-balanced',   engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'wasm',   decide: (out) => fairfaceProbWeighted(out, [1.0, 1.0, 0.5], 0.4) },
  { id: 'fairface-q8-prob-aggressive', engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'wasm',   decide: (out) => fairfaceProbWeighted(out, [1.5, 1.3, 0.8], 0.3) },
  { id: 'fairface-q8-margin',          engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'wasm',   decide: (out) => fairfaceMargin(out, -0.05) },
  { id: 'fairface-fp16-gpu-strict',    engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'fp16', device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-fp16-gpu-prob',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'fp16', device: 'webgpu', decide: (out) => fairfaceProbWeighted(out, [1.5, 1.3, 0.8], 0.3) },
  // === NEW: alternate WebGPU dtypes - hunting for fast+accurate sweet spot ===
  { id: 'fairface-fp32-gpu-strict',    engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'fp32', device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q8-gpu-strict',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q8',   device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-int8-gpu-strict',    engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'int8', device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  // === ROUND 2: WebGPU-specific 4-bit dtypes (recommended by transformers.js docs) ===
  { id: 'fairface-q4f16-gpu-strict',   engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4f16', device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-bnb4-gpu-strict',    engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'bnb4',  device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q4-gpu-strict',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4',    device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-uint8-gpu-strict',   engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'uint8', device: 'webgpu', decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-uint8-wasm-strict',  engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'uint8', device: 'wasm',   decide: (out) => strictKidFromBuckets(out[0].label) },
  // === ROUND 3: Batch parallelism on the q4-gpu winner ===
  { id: 'fairface-q4-gpu-batch2',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4',    device: 'webgpu', batchSize: 2, decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q4-gpu-batch4',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4',    device: 'webgpu', batchSize: 4, decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q4-gpu-batch8',      engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4',    device: 'webgpu', batchSize: 8, decide: (out) => strictKidFromBuckets(out[0].label) },
  { id: 'fairface-q4-gpu-batch16',     engine: 'transformers', model: 'onnx-community/fairface_age_image_detection-ONNX', task: 'image-classification', dtype: 'q4',    device: 'webgpu', batchSize: 16, decide: (out) => strictKidFromBuckets(out[0].label) },

  // ==================== AdamCodd YOLO11n face-age - 5 weighted variants ====================
  { id: 'yolo-fp32-strict',           engine: 'transformers', model: 'AdamCodd/yolo11n-face-age', task: 'object-detection', dtype: 'fp32', device: 'wasm',   decide: (out) => decideYoloFaceAge(out, 'strict') },
  { id: 'yolo-fp32-lenient',          engine: 'transformers', model: 'AdamCodd/yolo11n-face-age', task: 'object-detection', dtype: 'fp32', device: 'wasm',   decide: (out) => decideYoloFaceAge(out, 'lenient') },
  { id: 'yolo-fp32-vote',             engine: 'transformers', model: 'AdamCodd/yolo11n-face-age', task: 'object-detection', dtype: 'fp32', device: 'wasm',   decide: (out) => decideYoloFaceAge(out, 'vote') },
  { id: 'yolo-fp16-gpu-strict',       engine: 'transformers', model: 'AdamCodd/yolo11n-face-age', task: 'object-detection', dtype: 'fp16', device: 'webgpu', decide: (out) => decideYoloFaceAge(out, 'strict') },
  { id: 'yolo-fp16-gpu-lenient',      engine: 'transformers', model: 'AdamCodd/yolo11n-face-age', task: 'object-detection', dtype: 'fp16', device: 'webgpu', decide: (out) => decideYoloFaceAge(out, 'lenient') },

  // ==================== Human (TF.js) - 5 weighted variants ====================
  { id: 'human-wasm-thr14',           engine: 'human',  model: '@vladmandic/human@3.3.6', backend: 'wasm',  decide: (age) => age < 14 },
  { id: 'human-wasm-thr20',           engine: 'human',  model: '@vladmandic/human@3.3.6', backend: 'wasm',  decide: (age) => age < 20 },
  { id: 'human-wasm-thr30',           engine: 'human',  model: '@vladmandic/human@3.3.6', backend: 'wasm',  decide: (age) => age < 30 },
  { id: 'human-webgl-thr20',          engine: 'human',  model: '@vladmandic/human@3.3.6', backend: 'webgl', decide: (age) => age < 20 },
  { id: 'human-webgl-thr30',          engine: 'human',  model: '@vladmandic/human@3.3.6', backend: 'webgl', decide: (age) => age < 30 },

  // ==================== face-api (TF.js) - 5 weighted variants ====================
  { id: 'faceapi-tiny256-thr14',      engine: 'faceapi', model: '@vladmandic/face-api@1.7.15', detector: 'tiny', inputSize: 256, decide: (age) => age < 14 },
  { id: 'faceapi-tiny416-thr20',      engine: 'faceapi', model: '@vladmandic/face-api@1.7.15', detector: 'tiny', inputSize: 416, decide: (age) => age < 20 },
  { id: 'faceapi-ssd-thr14',          engine: 'faceapi', model: '@vladmandic/face-api@1.7.15', detector: 'ssd',  decide: (age) => age < 14 },
  { id: 'faceapi-ssd-thr20',          engine: 'faceapi', model: '@vladmandic/face-api@1.7.15', detector: 'ssd',  decide: (age) => age < 20 },
  { id: 'faceapi-ssd-thr30',          engine: 'faceapi', model: '@vladmandic/face-api@1.7.15', detector: 'ssd',  decide: (age) => age < 30 },

  // ==================== InsightFace genderage (1.3MB, super fast) - 5 thresholds ====================
  { id: 'insightface-thr14',          engine: 'insightface', model: 'public-data/insightface/buffalo_l/genderage.onnx', decide: (age) => age < 14 },
  { id: 'insightface-thr20',          engine: 'insightface', model: 'public-data/insightface/buffalo_l/genderage.onnx', decide: (age) => age < 20 },
  { id: 'insightface-thr30',          engine: 'insightface', model: 'public-data/insightface/buffalo_l/genderage.onnx', decide: (age) => age < 30 },
  { id: 'insightface-thr40',          engine: 'insightface', model: 'public-data/insightface/buffalo_l/genderage.onnx', decide: (age) => age < 40 },
  { id: 'insightface-thr50',          engine: 'insightface', model: 'public-data/insightface/buffalo_l/genderage.onnx', decide: (age) => age < 50 },
];

let manifest = null;
let lastReport = null;

async function loadManifest() {
  const r = await fetch('./manifest.json');
  manifest = await r.json();
  $('statKids').textContent = manifest.kids.length;
  $('statAdults').textContent = manifest.adults.length;
  $('statTotal').textContent = manifest.kids.length + manifest.adults.length;
  $('statVariants').textContent = VARIANTS.length;
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('image load failed: ' + src));
    img.src = src;
  });
}
async function loadImages() {
  const items = [];
  for (const p of manifest.kids) items.push({ path: p, label: 'kid' });
  for (const p of manifest.adults) items.push({ path: p, label: 'adult' });
  // Pre-load both RawImage (transformers.js) AND HTMLImageElement (Human/face-api)
  for (const it of items) {
    it.img = await RawImage.read(it.path);
    it.imgEl = await loadImageElement(it.path);
  }
  return items;
}

const LOAD_TIMEOUT_MS = 60_000;
const TIMEOUT_PREF_KEY = 'age-bench-timeout-sec';
function getTimeoutMs() {
  const el = document.getElementById('timeoutInput');
  const sec = el ? parseInt(el.value, 10) : NaN;
  if (Number.isFinite(sec) && sec >= 10) return sec * 1000;
  return 150_000;
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timeout (>${ms / 1000}s)`)), ms)),
  ]);
}

async function tryLoadTransformers(v) {
  const opts = { dtype: v.dtype };
  if (v.device) opts.device = v.device;
  const pipe = await withTimeout(pipeline(v.task, v.model, opts), LOAD_TIMEOUT_MS, 'load');
  return { type: 'transformers', pipe };
}

let humanModule = null;
async function tryLoadHuman(v) {
  if (!humanModule) {
    humanModule = await withTimeout(import('https://esm.sh/' + v.model), LOAD_TIMEOUT_MS, 'human module');
  }
  const Human = humanModule.default ?? humanModule.Human;
  const human = new Human({
    backend: v.backend,
    modelBasePath: 'https://vladmandic.github.io/human-models/models/',
    debug: false,
    cacheSensitivity: 0,
    filter: { enabled: false },
    face: {
      enabled: true,
      detector: { rotation: true, maxDetected: 5, minConfidence: 0.5 },
      mesh: { enabled: false },
      iris: { enabled: false },
      description: { enabled: true },
      emotion: { enabled: false },
      antispoof: { enabled: false },
      liveness: { enabled: false },
      gear: { enabled: false },
    },
    body: { enabled: false }, hand: { enabled: false }, object: { enabled: false }, segmentation: { enabled: false }, gesture: { enabled: false },
  });
  await withTimeout(human.load(), LOAD_TIMEOUT_MS, 'human.load');
  await withTimeout(human.warmup(), LOAD_TIMEOUT_MS, 'human.warmup');
  return { type: 'human', human };
}

let faceapiLoaded = null;
async function loadFaceApiScript(modelTag) {
  if (faceapiLoaded) return window.faceapi;
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = `https://cdn.jsdelivr.net/npm/${modelTag}/dist/face-api.js`;
    s.onload = resolve;
    s.onerror = () => reject(new Error('faceapi script failed to load'));
    document.head.appendChild(s);
  });
  faceapiLoaded = true;
  return window.faceapi;
}
async function tryLoadFaceApi(v) {
  const faceapi = await withTimeout(loadFaceApiScript(v.model), LOAD_TIMEOUT_MS, 'faceapi script');
  const modelUrl = 'https://cdn.jsdelivr.net/gh/vladmandic/face-api@master/model';
  if (v.detector === 'tiny') {
    if (!faceapi.nets.tinyFaceDetector.isLoaded) await faceapi.nets.tinyFaceDetector.loadFromUri(modelUrl);
  } else {
    if (!faceapi.nets.ssdMobilenetv1.isLoaded) await faceapi.nets.ssdMobilenetv1.loadFromUri(modelUrl);
  }
  if (!faceapi.nets.ageGenderNet.isLoaded) await faceapi.nets.ageGenderNet.loadFromUri(modelUrl);
  return { type: 'faceapi', faceapi, detector: v.detector };
}

let ortModule = null;
async function tryLoadInsightFace(v) {
  if (!ortModule) {
    ortModule = await withTimeout(import('https://esm.sh/onnxruntime-web@1.20.0'), LOAD_TIMEOUT_MS, 'onnxruntime-web');
    ortModule.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';
  }
  const url = `https://huggingface.co/${v.model.split('/')[0]}/${v.model.split('/')[1]}/resolve/main/models/${v.model.split('/').slice(2).join('/')}`;
  const session = await withTimeout(ortModule.InferenceSession.create(url, { executionProviders: ['wasm'] }), LOAD_TIMEOUT_MS, 'insightface session');
  return { type: 'insightface', session, ort: ortModule };
}

async function loadEngine(v) {
  if (v.engine === 'transformers') return await tryLoadTransformers(v);
  if (v.engine === 'human') return await tryLoadHuman(v);
  if (v.engine === 'faceapi') return await tryLoadFaceApi(v);
  if (v.engine === 'insightface') return await tryLoadInsightFace(v);
  throw new Error('unknown engine: ' + v.engine);
}

function preprocessForInsightFace(imgEl) {
  // InsightFace genderage expects 96x96 RGB, normalized to [-1, 1] (or per-channel mean subtraction)
  const canvas = document.createElement('canvas');
  canvas.width = 96; canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgEl, 0, 0, 96, 96);
  const imageData = ctx.getImageData(0, 0, 96, 96).data;
  const float32 = new Float32Array(3 * 96 * 96);
  // CHW format, RGB, normalized: pixel/127.5 - 1
  for (let i = 0; i < 96 * 96; i++) {
    float32[i] = imageData[i * 4] / 127.5 - 1;
    float32[96 * 96 + i] = imageData[i * 4 + 1] / 127.5 - 1;
    float32[2 * 96 * 96 + i] = imageData[i * 4 + 2] / 127.5 - 1;
  }
  return float32;
}

async function runInference(engine, v, item) {
  if (engine.type === 'transformers') {
    const out = await engine.pipe(item.img, { topk: 9 });
    return { isKid: !!v.decide(out), top: out[0] };
  }
  if (engine.type === 'human') {
    const result = await engine.human.detect(item.imgEl);
    const face = result.face[0];
    if (!face || typeof face.age !== 'number') return { isKid: true, top: { label: 'no-face', score: 0 } };
    return { isKid: !!v.decide(face.age), top: { label: `age=${face.age.toFixed(1)}`, score: face.score ?? 0 } };
  }
  if (engine.type === 'faceapi') {
    const inputSize = v.inputSize || 416;
    const opts = engine.detector === 'tiny'
      ? new engine.faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.4 })
      : new engine.faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 });
    const result = await engine.faceapi.detectSingleFace(item.imgEl, opts).withAgeAndGender();
    if (!result || typeof result.age !== 'number') return { isKid: true, top: { label: 'no-face', score: 0 } };
    return { isKid: !!v.decide(result.age), top: { label: `age=${result.age.toFixed(1)}`, score: result.detection?.score ?? 0 } };
  }
  if (engine.type === 'insightface') {
    const tensor = new engine.ort.Tensor('float32', preprocessForInsightFace(item.imgEl), [1, 3, 96, 96]);
    const inputName = engine.session.inputNames[0];
    const out = await engine.session.run({ [inputName]: tensor });
    const outName = engine.session.outputNames[0];
    const data = out[outName].data;
    // Output is typically [genderM, genderF, age_normalized*100] - but format varies
    // For InsightFace genderage: [gender_logits..., age]. Last value is age
    const age = data[data.length - 1] * 100;
    return { isKid: !!v.decide(age), top: { label: `age=${age.toFixed(1)}`, score: 0 } };
  }
  throw new Error('unknown engine type');
}

async function runVariant(v, items, idx, total) {
  setVariantState(v.id, 'running');
  markRunning(v);
  const remaining = total - idx;
  const dtypeStr = v.dtype || v.backend || v.detector || '';
  const deviceStr = v.device || v.backend || '';
  setBanner('running', `${idx} / ${total}`, v.id, `${v.model} · ${dtypeStr} · ${deviceStr} · טוען מודל...`, `<strong>${remaining - 1}</strong> וריאנטים נשארו`);
  log(`▶ ${v.id} [${v.engine}] (${v.model})`);
  const t0 = performance.now();
  let engine;
  try {
    engine = await loadEngine(v);
  } catch (err) {
    log(`  ✗ load failed: ${err.message}`);
    clearRunning();
    return { id: v.id, engine: v.engine, model: v.model, dtype: v.dtype || null, device: v.device || v.backend || null, loadFailed: true, error: String(err.message || err) };
  }
  const loadMs = performance.now() - t0;
  log(`  ✓ loaded in ${loadMs.toFixed(0)}ms`);
  // Warmup
  try { await runInference(engine, v, items[0]); } catch {}
  let kidCorrect = 0, kidTotal = 0, adultCorrect = 0, adultTotal = 0;
  const preds = [];
  let totalInferMs = 0;
  let timedOut = false;
  let processed = 0;
  const TIMEOUT_MS = getTimeoutMs();
  const inferStart = performance.now();
  const batchSize = v.batchSize || 1;
  for (let i = 0; i < items.length; i += batchSize) {
    if (performance.now() - inferStart > TIMEOUT_MS) {
      timedOut = true;
      log(`  ⏱ timeout after ${processed}/${items.length} images (>150s)`);
      break;
    }
    const chunk = items.slice(i, i + batchSize);
    setProgress(i + chunk.length, items.length);
    const remaining = total - idx;
    setBanner('running', `${idx} / ${total}`, v.id, `${v.model} · ${dtypeStr} · ${deviceStr}${batchSize > 1 ? ' · batch=' + batchSize : ''}`, `<strong>${remaining - 1}</strong> וריאנטים נשארו`);
    const t = performance.now();
    let chunkResults;
    try {
      // Run chunk concurrently via Promise.all - lets the GPU/worker scheduler pipeline them
      chunkResults = await Promise.all(chunk.map((it) => runInference(engine, v, it)));
    } catch (err) {
      log(`  ⚠ batch inference error: ${err.message}`);
      for (const it of chunk) preds.push({ path: it.path, true: it.label, pred: 'error', ms: 0 });
      continue;
    }
    const chunkMs = performance.now() - t;
    totalInferMs += chunkMs;
    chunk.forEach((it, j) => {
      processed++;
      const inferRes = chunkResults[j];
      const pred = inferRes.isKid ? 'kid' : 'adult';
      if (it.label === 'kid') { kidTotal++; if (pred === 'kid') kidCorrect++; }
      else { adultTotal++; if (pred === 'adult') adultCorrect++; }
      preds.push({ path: it.path, true: it.label, pred, top: inferRes.top, ms: chunkMs / chunk.length });
    });
  }
  const denom = Math.max(1, processed);
  clearRunning();
  return {
    id: v.id, model: v.model, dtype: v.dtype, device: v.device,
    loadFailed: false,
    timedOut,
    processed,
    loadMs,
    avgInferMs: totalInferMs / denom,
    totalInferMs,
    kidAccuracy: kidTotal ? kidCorrect / kidTotal : 0,
    adultAccuracy: adultTotal ? adultCorrect / adultTotal : 0,
    overallAccuracy: (kidCorrect + adultCorrect) / Math.max(1, items.length),
    kidCorrect, kidTotal, adultCorrect, adultTotal,
    preds,
  };
}

function computeScore(r) {
  if (r.loadFailed || r.timedOut) return null;
  const speedFactor = Math.max(0, 1 - r.avgInferMs / 2000);
  return r.overallAccuracy * 0.85 + speedFactor * 0.15;
}

function rankBadge(i) {
  if (i === 0) return '<span class="rank gold">1</span>';
  if (i === 1) return '<span class="rank silver">2</span>';
  if (i === 2) return '<span class="rank bronze">3</span>';
  return `<span class="rank">${i + 1}</span>`;
}
function accCell(correct, total, ratio) {
  const pct = ratio * 100;
  const cls = pct >= 85 ? 'acc-good' : pct >= 60 ? 'acc-mid' : 'acc-bad';
  return `<div class="acc-cell"><span><strong>${correct}/${total}</strong> · ${pct.toFixed(0)}%</span><div class="acc-bar"><div class="${cls}" style="width:${pct}%"></div></div></div>`;
}

function renderResults(results) {
  const tbody = $('resultsBody');
  tbody.innerHTML = '';
  if (results.length === 0) { $('resultsTable').hidden = true; $('emptyResults').hidden = false; return; }
  $('resultsTable').hidden = false;
  $('emptyResults').hidden = true;
  const enriched = results.map(r => ({ ...r, score: computeScore(r) }));
  const sorted = enriched.slice().sort((a, b) => {
    if (a.score == null && b.score == null) return 0;
    if (a.score == null) return 1;
    if (b.score == null) return -1;
    return b.score - a.score;
  });
  let bestScore = -1;
  for (const r of sorted) if (r.score != null && r.score > bestScore) bestScore = r.score;
  const variantById = Object.fromEntries(VARIANTS.map(v => [v.id, v]));
  sorted.forEach((r, i) => {
    const tr = document.createElement('tr');
    const modelCell = `<div class="model-cell"><span class="model-name">${r.model.split('/')[1] || r.model}</span><span class="model-meta">${r.model.split('/')[0] || ''}</span></div>`;
    const tagSuffix = r.id.split('-').slice(-1)[0];
    const opts = `<span class="tag">${r.dtype}</span> <span class="tag">${r.device}</span>${tagSuffix && tagSuffix !== r.dtype && tagSuffix !== r.device ? ` <span class="tag">${tagSuffix}</span>` : ''}`;
    const v = variantById[r.id];
    const sizeCell = v ? formatSizeMB(getCachedSizeMB(v)) : '<span class="model-meta">—</span>';
    if (r.loadFailed) {
      tr.classList.add('fail');
      tr.innerHTML = `<td>${rankBadge(i)}</td><td>—</td><td>${modelCell}</td><td>${sizeCell}</td><td>${opts}</td><td colspan="5"><span class="err">✗ load failed:</span> ${r.error || ''}</td>`;
    } else if (r.timedOut) {
      tr.classList.add('fail');
      tr.innerHTML = `<td>${rankBadge(i)}</td><td>—</td><td>${modelCell}</td><td>${sizeCell}</td><td>${opts}</td><td colspan="5"><span class="err">⏱ timeout</span> אחרי ${r.processed} תמונות, avg ${r.avgInferMs.toFixed(0)}ms</td>`;
    } else {
      if (Math.abs(r.score - bestScore) < 1e-9) tr.classList.add('best');
      const score100 = (r.score * 100).toFixed(1);
      const scoreCell = `<div class="score"><span class="score-num">${score100}</span><div class="score-bar"><div style="width:${r.score * 100}%"></div></div></div>`;
      tr.innerHTML = `<td>${rankBadge(i)}</td><td>${scoreCell}</td><td>${modelCell}</td><td>${sizeCell}</td><td>${opts}</td>`
        + `<td>${accCell(r.kidCorrect + r.adultCorrect, r.kidTotal + r.adultTotal, r.overallAccuracy)}</td>`
        + `<td>${accCell(r.kidCorrect, r.kidTotal, r.kidAccuracy)}</td>`
        + `<td>${accCell(r.adultCorrect, r.adultTotal, r.adultAccuracy)}</td>`
        + `<td><strong>${r.avgInferMs.toFixed(0)}</strong> ms</td>`
        + `<td>${(r.totalInferMs / 1000).toFixed(1)} s</td>`;
    }
    tbody.appendChild(tr);
  });
}

const CACHE_KEY = 'age-bench-results-v5';
const RUNNING_KEY = 'age-bench-running-v5';
const SIZE_CACHE_KEY = 'age-bench-sizes-v1';

const TRANSFORMERS_DTYPE_TO_FILE = {
  fp32: 'model.onnx',
  fp16: 'model_fp16.onnx',
  q8: 'model_quantized.onnx',
  int8: 'model_int8.onnx',
  uint8: 'model_uint8.onnx',
  q4: 'model_q4.onnx',
  q4f16: 'model_q4f16.onnx',
  bnb4: 'model_bnb4.onnx',
};

// Hardcoded sizes for engines whose loaders pull a bundle of files.
// Values in MB, derived from the actual file sizes downloaded by each library.
const HARDCODED_SIZES_MB = {
  'human-wasm': 6.0,    // face-detection + age-gender + emotion bundle
  'human-webgl': 6.0,
  'faceapi-tiny': 0.6,  // tiny_face_detector (~190KB) + age_gender (~420KB)
  'faceapi-ssd': 6.2,   // ssd_mobilenetv1 (~5.8MB) + age_gender (~420KB)
};

function loadSizeCache() {
  try { return JSON.parse(localStorage.getItem(SIZE_CACHE_KEY) || '{}'); } catch { return {}; }
}
function saveSizeCache(cache) {
  try { localStorage.setItem(SIZE_CACHE_KEY, JSON.stringify(cache)); } catch {}
}

async function headSize(url) {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    if (!r.ok) return null;
    const len = r.headers.get('content-length');
    return len ? parseInt(len, 10) : null;
  } catch { return null; }
}

function variantSizeKey(v) {
  if (v.engine === 'transformers') return `tf|${v.model}|${v.dtype}`;
  if (v.engine === 'insightface') return `if|${v.model}`;
  if (v.engine === 'human') return `human-${v.backend}`;
  if (v.engine === 'faceapi') return `faceapi-${v.detector}`;
  return v.id;
}

async function fetchVariantSizeMB(v) {
  if (v.engine === 'human') return HARDCODED_SIZES_MB[`human-${v.backend}`] ?? null;
  if (v.engine === 'faceapi') return HARDCODED_SIZES_MB[`faceapi-${v.detector}`] ?? null;
  if (v.engine === 'transformers') {
    const file = TRANSFORMERS_DTYPE_TO_FILE[v.dtype];
    if (!file) return null;
    const url = `https://huggingface.co/${v.model}/resolve/main/onnx/${file}`;
    const bytes = await headSize(url);
    return bytes ? bytes / (1024 * 1024) : null;
  }
  if (v.engine === 'insightface') {
    const parts = v.model.split('/');
    const url = `https://huggingface.co/${parts[0]}/${parts[1]}/resolve/main/models/${parts.slice(2).join('/')}`;
    const bytes = await headSize(url);
    return bytes ? bytes / (1024 * 1024) : null;
  }
  return null;
}

let sizeCache = loadSizeCache();
async function getVariantSizeMB(v) {
  const key = variantSizeKey(v);
  if (key in sizeCache) return sizeCache[key];
  const mb = await fetchVariantSizeMB(v);
  sizeCache[key] = mb;
  saveSizeCache(sizeCache);
  return mb;
}
function getCachedSizeMB(v) {
  const key = variantSizeKey(v);
  return key in sizeCache ? sizeCache[key] : null;
}
function formatSizeMB(mb) {
  if (mb == null) return '<span class="model-meta">—</span>';
  if (mb < 1) return `<span class="tag">${(mb * 1024).toFixed(0)} KB</span>`;
  if (mb < 100) return `<span class="tag">${mb.toFixed(1)} MB</span>`;
  return `<span class="tag">${mb.toFixed(0)} MB</span>`;
}
async function ensureAllVariantSizes() {
  const tasks = VARIANTS.map(async (v) => {
    if (variantSizeKey(v) in sizeCache) return;
    await getVariantSizeMB(v);
  });
  await Promise.all(tasks);
}
function saveCache(variant, result) {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    const map = raw ? JSON.parse(raw) : {};
    map[variant.id] = { ...result, savedAt: Date.now() };
    localStorage.setItem(CACHE_KEY, JSON.stringify(map));
  } catch (e) { console.warn('cache save failed', e); }
}
function loadCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch { return {}; }
}
function clearCache() {
  localStorage.removeItem(CACHE_KEY);
  localStorage.removeItem(RUNNING_KEY);
  localStorage.removeItem(SIZE_CACHE_KEY);
  sizeCache = {};
}
function markRunning(variant) {
  try { localStorage.setItem(RUNNING_KEY, JSON.stringify({ id: variant.id, model: variant.model, dtype: variant.dtype, device: variant.device, startedAt: Date.now() })); } catch {}
}
function clearRunning() {
  localStorage.removeItem(RUNNING_KEY);
}
function detectCrashedFromLastRun() {
  // If RUNNING_KEY is set on page load, the page crashed mid-variant. Mark that variant as 'crashed' in cache.
  try {
    const raw = localStorage.getItem(RUNNING_KEY);
    if (!raw) return null;
    const info = JSON.parse(raw);
    const cache = loadCache();
    if (cache[info.id]) { localStorage.removeItem(RUNNING_KEY); return null; }
    const crashed = { id: info.id, model: info.model, dtype: info.dtype, device: info.device, loadFailed: true, error: 'page crashed during run (likely OOM or WASM trap) - skipped' };
    cache[info.id] = { ...crashed, savedAt: Date.now() };
    localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
    localStorage.removeItem(RUNNING_KEY);
    return crashed;
  } catch { return null; }
}

$('runBtn').addEventListener('click', async () => {
  $('runBtn').disabled = true;
  $('downloadBtn').disabled = true;
  $('log').textContent = '';
  setBanner('running', '...', 'טוען תמונות', 'מכין את כל התמונות ב-RAM לפני שמתחילים', '');
  await loadManifest();
  await ensureAllVariantSizes();
  const items = await loadImages();
  log(`Loaded ${items.length} images (${manifest.kids.length} kids, ${manifest.adults.length} adults)`);
  const cache = loadCache();
  const results = [];
  // Seed results + chip states for cached variants
  for (const v of VARIANTS) {
    if (cache[v.id]) {
      results.push(cache[v.id]);
      setVariantState(v.id, cache[v.id].loadFailed ? 'fail' : cache[v.id].timedOut ? 'timeout' : 'cached');
    } else {
      setVariantState(v.id, 'queued');
    }
  }
  if (results.length) { log(`📦 Loaded ${results.length} cached variants from localStorage`); renderResults(results); }
  for (let v = 0; v < VARIANTS.length; v++) {
    const variant = VARIANTS[v];
    if (cache[variant.id]) {
      log(`⏭ ${variant.id} - using cached result`);
      continue;
    }
    const r = await runVariant(variant, items, v + 1, VARIANTS.length);
    if (r.loadFailed) { log(`  ⚠ ${r.error}`); setVariantState(variant.id, 'fail'); }
    else if (r.timedOut) { log(`  ⏱ timed out`); setVariantState(variant.id, 'timeout'); }
    else { log(`  acc=${(r.overallAccuracy*100).toFixed(1)}%  kid=${r.kidCorrect}/${r.kidTotal}  adult=${r.adultCorrect}/${r.adultTotal}  avg=${r.avgInferMs.toFixed(0)}ms`); setVariantState(variant.id, 'done'); }
    results.push(r);
    saveCache(variant, r);
    renderResults(results);
  }
  setProgress(0, 0);
  const summary = results.filter(r => !r.loadFailed && !r.timedOut);
  const best = summary.slice().sort((a, b) => (computeScore(b) ?? 0) - (computeScore(a) ?? 0))[0];
  setBanner('idle', 'סיום', 'ההרצה הסתיימה', `${results.length} וריאנטים נבדקו · ${results.length - summary.length} נכשלו או נחתכו`, best ? `הזוכה: <strong>${best.id}</strong>` : '');
  lastReport = { generatedAt: new Date().toISOString(), datasetSize: items.length, kidsCount: manifest.kids.length, adultsCount: manifest.adults.length, results };
  $('downloadBtn').disabled = false;
  $('runBtn').disabled = false;
});

$('clearBtn').addEventListener('click', () => {
  if (!confirm('לנקות את כל התוצאות השמורות?')) return;
  clearCache();
  $('resultsBody').innerHTML = '';
  $('resultsTable').hidden = true;
  $('emptyResults').hidden = false;
  $('log').textContent = 'Cache cleared.\n';
  for (const v of VARIANTS) variantState[v.id] = 'queued';
  renderStrip();
  refreshVariantProgress();
  setBanner('idle', 'ממתין', 'מוכן להרצה חדשה', 'cache נוקה - הכל יירוץ מאפס', '');
});

// Restore + persist timeout preference
(() => {
  const el = document.getElementById('timeoutInput');
  if (!el) return;
  const saved = localStorage.getItem(TIMEOUT_PREF_KEY);
  if (saved && Number.isFinite(parseInt(saved, 10))) el.value = parseInt(saved, 10);
  el.addEventListener('change', () => {
    const v = parseInt(el.value, 10);
    if (Number.isFinite(v) && v >= 10) localStorage.setItem(TIMEOUT_PREF_KEY, String(v));
  });
})();

// On page load, render any cached results + initialize chip strip
(async () => {
  for (const v of VARIANTS) variantState[v.id] = 'queued';
  renderStrip();
  await loadManifest();
  const crashed = detectCrashedFromLastRun();
  if (crashed) log(`💥 שיחזרתי קריסה מהריצה הקודמת: ${crashed.id} סומן כ-fail ולא יורץ שוב`);
  const cache = loadCache();
  for (const v of VARIANTS) {
    if (cache[v.id]) {
      variantState[v.id] = cache[v.id].loadFailed ? 'fail' : cache[v.id].timedOut ? 'timeout' : 'cached';
    }
  }
  renderStrip();
  refreshVariantProgress();
  const cached = Object.values(cache);
  if (cached.length) {
    log(`📦 ${cached.length} variants restored from localStorage`);
    renderResults(cached);
    lastReport = { generatedAt: new Date().toISOString(), restoredFromCache: true, results: cached };
    $('downloadBtn').disabled = false;
    setBanner('idle', 'ממתין', `${cached.length} וריאנטים שמורים מהפעם הקודמת`, 'הרץ benchmark כדי לסיים את השאר', '');
  }
  // Fetch model sizes in the background, re-render when done
  ensureAllVariantSizes().then(() => {
    if (cached.length) renderResults(cached);
  });
})();

$('downloadBtn').addEventListener('click', () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'bench-report.json';
  a.click();
});
