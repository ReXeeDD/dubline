/* Dubline - local dubbing studio front end */
'use strict';

const view = document.getElementById('view');
const modalRoot = document.getElementById('modal-root');
const toastRoot = document.getElementById('toast-root');

let VOICES = [];
let SEARCH = '';
let poller = null;

/* ------------------------------------------------------------- helpers --- */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const jsonPost = (path, body, method = 'POST') => api(path, {
  method, headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

function toast(msg, isError = false) {
  const el = document.createElement('div');
  el.className = 'toast' + (isError ? ' err' : '');
  el.textContent = msg;
  toastRoot.append(el);
  setTimeout(() => el.remove(), isError ? 6000 : 3200);
}

function hhmmss(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
}

function ago(ts) {
  const d = Date.now() / 1000 - ts;
  const steps = [[31536000, 'year'], [2592000, 'month'], [86400, 'day'],
    [3600, 'hour'], [60, 'minute']];
  for (const [secs, name] of steps) {
    if (d >= secs) { const n = Math.floor(d / secs); return `${n} ${name}${n > 1 ? 's' : ''} ago`; }
  }
  return 'just now';
}

/* --------------------------------------------------------------- modal --- */
function modal({ title, body, footer, onMount, wide }) {
  modalRoot.innerHTML = `
    <div class="overlay">
      <div class="modal"${wide ? ' style="width:min(760px,100%)"' : ''}>
        <header>
          <h2>${esc(title)}</h2>
          <button class="btn icon ghost" data-close>&#10005;</button>
        </header>
        <div class="content">${body}</div>
        ${footer ? `<footer>${footer}</footer>` : ''}
      </div>
    </div>`;
  const overlay = modalRoot.querySelector('.overlay');
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });
  modalRoot.querySelector('[data-close]').onclick = closeModal;
  document.addEventListener('keydown', escClose);
  if (onMount) onMount(modalRoot);
}
function escClose(e) { if (e.key === 'Escape') closeModal(); }
function closeModal() {
  modalRoot.innerHTML = '';
  document.removeEventListener('keydown', escClose);
}

/* -------------------------------------------------------------- voices --- */
async function loadVoices() {
  if (VOICES.length) return VOICES;
  try { VOICES = await api('/api/voices'); } catch (e) { toast(e.message, true); }
  return VOICES;
}

function voiceOptions(selected) {
  const groups = {};
  for (const v of VOICES) (groups[v.locale] ||= []).push(v);
  const names = { clone: 'Your cloned voices',
    'en-US': 'United States', 'en-GB': 'United Kingdom',
    'en-AU': 'Australia', 'en-CA': 'Canada', 'en-IE': 'Ireland',
    'en-IN': 'India', 'en-NZ': 'New Zealand', 'en-ZA': 'South Africa',
    'en-NG': 'Nigeria', 'en-KE': 'Kenya', 'en-TZ': 'Tanzania',
    'en-PH': 'Philippines', 'en-SG': 'Singapore', 'en-HK': 'Hong Kong' };
  return Object.entries(groups).map(([loc, list]) => `
    <optgroup label="${esc(names[loc] || loc)}">
      ${list.map((v) => `<option value="${esc(v.id)}"${v.id === selected ? ' selected' : ''}>
        ${esc(v.name)} - ${esc(v.gender)}${v.clone ? ' - cloned' : ''}${v.multilingual ? ' - multilingual' : ''}
      </option>`).join('')}
    </optgroup>`).join('');
}

let previewAudio = null;
let previewSeq = 0;

async function speak({ voice, pitch = 0, speed = 0, volume = 0, soften = 0, text }) {
  if (previewAudio) { previewAudio.pause(); previewAudio = null; }
  const mine = ++previewSeq;
  const res = await fetch('/api/voices/preview', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice, pitch, speed, volume, soften, text }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'preview failed');
  const blob = await res.blob();
  if (mine !== previewSeq) return;        // a newer request already won
  const url = URL.createObjectURL(blob);
  previewAudio = new Audio(url);
  previewAudio.onended = () => URL.revokeObjectURL(url);
  await previewAudio.play();
}

async function previewVoice(voice, btn, tune = {}) {
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Loading...';
  try { await speak({ voice, ...tune }); } catch (e) { toast(e.message, true); }
  btn.disabled = false; btn.textContent = label;
}

/* The three things the speech engine can actually change. Every slider here
   maps onto a real edge-tts prosody parameter, so what you hear in the sample
   is exactly what the finished dub does. */
const TUNER_KNOBS = [
  { key: 'pitch', label: 'Voice depth', min: -50, max: 30, step: 2, unit: 'Hz',
    ends: ['deeper', 'normal', 'higher'] },
  { key: 'speed', label: 'Speaking pace', min: -40, max: 50, step: 5, unit: '%',
    ends: ['slower', 'normal', 'faster'] },
  { key: 'volume', label: 'Loudness', min: -50, max: 50, step: 5, unit: '%',
    ends: ['quieter', 'normal', 'louder'] },
  // Cloned voices only: drives the spectral cleaner inside the conversion.
  { key: 'soften', label: 'Softness', min: 0, max: 100, step: 5, unit: '',
    ends: ['raw', '', 'smoothest'], cloneOnly: true, zero: 'off' },
];

const isCloneVoice = (id) => VOICES.some((v) => v.id === id && v.clone);

/* Measured on the Rei model: past ~45 the cleaner starts eroding the things
   that make the voice recognisably his, so the slider says so instead of
   pretending the trade does not exist. */
function softNote(v) {
  if (v === 0) return 'Off - the conversion is left exactly as the model produces it.';
  if (v <= 45) return 'Smoother, and still clearly the same voice.';
  if (v <= 70) return 'Noticeably smoother. Starts drifting from the original voice.';
  return 'Very smooth, but it stops sounding like him. Worth it only if the roughness is unbearable.';
}

const fmtKnob = (k, v) => (v === 0 ? (k.zero || 'normal')
  : (k.unit ? (v > 0 ? '+' : '') + v + ' ' + k.unit : String(v)));

function voiceTuner({ value = {}, getVoice, sample }) {
  const val = (k) => +(value[k.key] ?? 0);

  const html = `
    <div class="field">
      ${TUNER_KNOBS.map((k) => `
        <div class="knobrow${k.cloneOnly ? ' clone-only' : ''}" data-for="${k.key}">
        <label for="tk_${k.key}" style="margin-top:10px">${k.label}
          <b id="tv_${k.key}" style="color:var(--accent)">${fmtKnob(k, val(k))}</b>
        </label>
        <input type="range" class="tuneknob" data-key="${k.key}" id="tk_${k.key}"
               min="${k.min}" max="${k.max}" step="${k.step}" value="${val(k)}"
               style="width:100%">
        <div style="display:flex;justify-content:space-between;color:var(--text-3);font-size:11.5px;margin-top:2px">
          ${k.ends.map((e) => `<span>${e}</span>`).join('')}
        </div>
        ${k.key === 'soften' ? `<div class="hint" id="softnote" style="margin:4px 0 0">${softNote(val(k))}</div>` : ''}
        </div>`).join('')}
      <div style="display:flex;gap:8px;align-items:center;margin-top:12px">
        <button class="btn ghost" id="playpitch" style="padding:5px 12px;font-size:13px">Play sample</button>
        <button class="btn ghost" id="resetknobs" style="padding:5px 12px;font-size:13px">Reset</button>
        <span class="hint" id="pitchnote" style="margin:0">Drag a slider - the sample replays automatically.</span>
      </div>
    </div>`;

  function mount(root) {
    const sliders = [...root.querySelectorAll('.tuneknob')];
    const note = root.querySelector('#pitchnote');
    const read = () => Object.fromEntries(sliders.map((s) => [s.dataset.key, +s.value]));
    let timer = null;

    const show = () => {
      sliders.forEach((s) => {
        const k = TUNER_KNOBS.find((x) => x.key === s.dataset.key);
        root.querySelector(`#tv_${k.key}`).textContent = fmtKnob(k, +s.value);
      });
      const sn = root.querySelector('#softnote');
      if (sn) sn.textContent = softNote(read().soften ?? 0);
    };

    // Softness lives inside the conversion, so it only means anything for a
    // cloned voice. Hide it rather than show a control that does nothing.
    const syncClone = () => {
      const on = isCloneVoice(getVoice());
      root.querySelectorAll('.knobrow.clone-only')
        .forEach((r) => r.classList.toggle('hidden', !on));
    };

    const fire = async () => {
      const clone = isCloneVoice(getVoice());
      note.textContent = clone ? 'Converting...' : 'Speaking...';
      try {
        await speak({ voice: getVoice(), text: sample, ...read() });
        note.textContent = 'Drag a slider - the sample replays automatically.';
      } catch (e) { note.textContent = e.message; }
    };

    sliders.forEach((s) => {
      s.oninput = () => {
        show();
        clearTimeout(timer);
        // A cloned voice re-converts on the GPU, so give the drag longer to
        // settle before spending that.
        timer = setTimeout(fire, isCloneVoice(getVoice()) ? 700 : 450);
      };
    });
    syncClone();
    root.querySelectorAll('select').forEach((sel) => {
      sel.addEventListener('change', syncClone);
    });
    root.querySelector('#playpitch').onclick = (e) => { e.preventDefault(); fire(); };
    root.querySelector('#resetknobs').onclick = (e) => {
      e.preventDefault();
      sliders.forEach((s) => { s.value = 0; });
      show();
      clearTimeout(timer);
      timer = setTimeout(fire, 200);
    };
    show();
    return read;
  }

  return { html, mount };
}

/* -------------------------------------------------------------- upload --- */
const LANGS = [['zh', 'Chinese (Mandarin)'], ['ja', 'Japanese'], ['ko', 'Korean'],
  ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'], ['ru', 'Russian'],
  ['ar', 'Arabic'], ['hi', 'Hindi'], ['pt', 'Portuguese'], ['it', 'Italian'],
  ['auto', 'Detect automatically']];

async function openUpload() {
  await loadVoices();
  const cfg = await api('/api/settings').catch(() => ({}));

  modal({
    title: 'Upload a video to dub',
    body: `
      ${cfg.has_key ? '' : `<div class="alert">No Groq API key set yet.
        Add it in Settings before uploading, or processing will fail.</div>`}
      <div class="drop" id="drop">
        <input type="file" id="file" accept="video/*" multiple hidden>
        <b>Choose videos or drag them here</b>
        <span>MP4, MKV, WEBM, MOV, AVI and more - pick several to queue a batch</span>
      </div>
      <div class="field" style="margin-top:18px">
        <label for="title">Title</label>
        <input type="text" id="title" placeholder="Taken from the filename">
      </div>
      <div class="row">
        <div class="field">
          <label for="lang">Spoken language</label>
          <select id="lang">${LANGS.map(([c, n]) =>
            `<option value="${c}"${c === (cfg.source_language || 'zh') ? ' selected' : ''}>${n}</option>`).join('')}</select>
        </div>
        <div class="field">
          <label for="voice">English dub voice</label>
          <select id="voice">${voiceOptions(cfg.voice)}</select>
          <div class="hint"><button class="btn ghost" id="prev" style="padding:4px 10px;font-size:12px;margin-top:6px">Preview voice</button></div>
        </div>
      </div>
      <div id="upstate"></div>`,
    footer: `<button class="btn ghost" data-cancel>Cancel</button>
             <button class="btn primary" id="go" disabled>Start dubbing</button>`,
    onMount(root) {
      const drop = root.querySelector('#drop');
      const fileInput = root.querySelector('#file');
      const go = root.querySelector('#go');
      const titleInput = root.querySelector('#title');
      let chosen = [];

      // Several files queue behind each other: only one video is processed at a
      // time anyway, so choosing a whole season and walking away is the useful
      // shape. A batch takes each file's own name - one Title box cannot name
      // seven episodes.
      const pick = (files) => {
        chosen = [...(files || [])].filter((f) => f && f.size);
        if (!chosen.length) return;
        drop.classList.add('has-file');
        const mb = chosen.reduce((n, f) => n + f.size, 0) / 1048576;
        drop.innerHTML = chosen.length === 1
          ? `<b>${esc(chosen[0].name)}</b><span>${mb.toFixed(1)} MB - click to change</span>`
          : `<b>${chosen.length} videos</b><span>${mb.toFixed(0)} MB total - `
            + `they will be dubbed one after another</span>`;
        drop.append(fileInput);
        titleInput.disabled = chosen.length > 1;
        titleInput.placeholder = chosen.length > 1
          ? 'Each video keeps its own filename' : 'Taken from the filename';
        if (chosen.length > 1) titleInput.value = '';
        else if (!titleInput.value) titleInput.value = chosen[0].name.replace(/\.[^.]+$/, '');
        go.textContent = chosen.length > 1
          ? `Start dubbing ${chosen.length} videos` : 'Start dubbing';
        go.disabled = false;
      };

      drop.onclick = () => fileInput.click();
      fileInput.onchange = () => pick(fileInput.files);
      ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
        e.preventDefault(); drop.classList.add('over');
      }));
      ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
        e.preventDefault(); drop.classList.remove('over');
      }));
      drop.addEventListener('drop', (e) => pick(e.dataTransfer.files));

      root.querySelector('#prev').onclick = (e) => {
        e.preventDefault();
        previewVoice(root.querySelector('#voice').value, e.target);
      };
      root.querySelector('[data-cancel]').onclick = closeModal;

      // One file at a time, in order. Sending them all at once would have the
      // browser split the upstream between them, so nothing finishes early and
      // the queue cannot start on the first video until the last has landed.
      const sendOne = (file, index) => new Promise((resolve, reject) => {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('title', chosen.length > 1 ? '' : titleInput.value);
        fd.append('voice', root.querySelector('#voice').value);
        fd.append('source_language', root.querySelector('#lang').value);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload');
        xhr.upload.onprogress = (e) => {
          if (!e.lengthComputable) return;
          const pct = Math.round((e.loaded / e.total) * 100);
          barFill.style.width = pct + '%';
          const of = chosen.length > 1 ? ` - file ${index + 1} of ${chosen.length}` : '';
          stage.textContent = `Uploading ${esc(file.name).slice(0, 40)}${of}`
            + ` ${pct}% (${(e.loaded / 1048576).toFixed(0)}`
            + ` of ${(e.total / 1048576).toFixed(0)} MB)`;
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) return resolve();
          let msg = 'Upload failed';
          try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (e) { /* */ }
          reject(new Error(`${file.name}: ${msg}`));
        };
        xhr.onerror = () => reject(new Error('Upload failed - is the server still running?'));
        xhr.send(fd);
      });

      let barFill; let stage;
      go.onclick = async () => {
        if (!chosen.length) return;
        const state = root.querySelector('#upstate');
        go.disabled = true;
        root.querySelector('[data-cancel]').disabled = true;
        state.innerHTML = `<div class="progress-wrap"><div class="bar"><i style="width:0%"></i></div>
          <div class="stage">Uploading...</div></div>`;
        barFill = state.querySelector('.bar > i');
        stage = state.querySelector('.stage');

        const failed = [];
        for (let i = 0; i < chosen.length; i++) {
          try {
            await sendOne(chosen[i], i);
          } catch (err) {
            failed.push(err.message);
          }
        }

        if (failed.length === chosen.length) {
          state.innerHTML = `<div class="alert">${esc(failed[0])}</div>`;
          go.disabled = false;
          root.querySelector('[data-cancel]').disabled = false;
          return;
        }
        closeModal();
        const sent = chosen.length - failed.length;
        toast(failed.length
          ? `${sent} of ${chosen.length} uploaded - ${failed.length} failed`
          : (sent > 1 ? `${sent} videos queued for dubbing`
                      : 'Upload complete - dubbing started'), !!failed.length);
        location.hash = '#/';
        render();
      };
    },
  });
}

/* -------------------------------------------------------- download --- */
/* Paste a link, see what it offers, pick one. Nothing is dubbed here: the
   video lands on the Downloads shelf and waits to be sent for translation. */
function openDownload() {
  modal({
    title: 'Download from a link',
    body: `
      <div class="field">
        <label for="url">Video link</label>
        <input type="text" id="url" placeholder="https://..." autocomplete="off" spellcheck="false">
        <div class="hint">Paste the address of the video page.</div>
      </div>
      <div id="dlinfo"></div>`,
    footer: `<button class="btn ghost" data-cancel>Cancel</button>
             <button class="btn primary" id="fetch">Find qualities</button>`,
    onMount(root) {
      const urlBox = root.querySelector('#url');
      const info = root.querySelector('#dlinfo');
      const btn = root.querySelector('#fetch');
      root.querySelector('[data-cancel]').onclick = closeModal;
      setTimeout(() => urlBox.focus(), 60);

      let found = null;

      const look = async () => {
        const url = urlBox.value.trim();
        if (!url) return;
        btn.disabled = true;
        btn.textContent = 'Looking...';
        info.innerHTML = '<div class="hint" style="padding:14px 0">Reading the link - this takes a few seconds.</div>';
        try {
          found = await jsonPost('/api/download/probe', { url });
        } catch (e) {
          info.innerHTML = `<div class="alert">${esc(e.message)}</div>`;
          btn.disabled = false;
          btn.textContent = 'Find qualities';
          return;
        }
        if (!found.qualities.length) {
          info.innerHTML = '<div class="alert">No downloadable video streams at that link.</div>';
          btn.disabled = false;
          btn.textContent = 'Find qualities';
          return;
        }
        // The largest picture is preselected; it is nearly always what is
        // wanted, and the rest are one click away.
        info.innerHTML = `
          <div class="found">
            <b>${esc(found.title)}</b>
            <span>${found.duration ? hhmmss(found.duration) : ''}${
              found.uploader ? ` &middot; ${esc(found.uploader)}` : ''}</span>
          </div>
          <div class="quals">
            ${found.qualities.map((q, i) => `
              <label class="qual">
                <input type="radio" name="q" value="${i}"${i === 0 ? ' checked' : ''}>
                <b>${q.label}</b>
                <span>${esc(q.vcodec)} &middot; ${q.abr}k audio &middot; .${q.ext}</span>
                <em>${q.size ? (q.size / 1048576).toFixed(0) + ' MB' : '?'}</em>
                <code>${esc(q.format)}</code>
              </label>`).join('')}
          </div>`;
        btn.disabled = false;
        btn.textContent = 'Download';
        btn.onclick = start;
      };

      const start = async () => {
        const pickIdx = +root.querySelector('input[name="q"]:checked').value;
        const q = found.qualities[pickIdx];
        btn.disabled = true;
        try {
          await jsonPost('/api/download', {
            url: urlBox.value.trim(), format: q.format, title: found.title,
          });
        } catch (e) {
          info.innerHTML += `<div class="alert">${esc(e.message)}</div>`;
          btn.disabled = false;
          return;
        }
        closeModal();
        toast(`Downloading ${q.label} - it lands in Downloads`);
        location.hash = '#/downloads';
        render();
      };

      btn.onclick = look;
      urlBox.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); btn.click(); } };
    },
  });
}

/* ------------------------------------------------------------ settings --- */
/* The panel opens immediately with a placeholder and fills in as data lands.
   Waiting on the voice list and the model list before rendering made the
   button look dead for several seconds. */
function openSettings() {
  modal({
    wide: true,
    title: 'Settings',
    body: `<div id="settings-body" style="display:grid;place-items:center;
             gap:14px;padding:46px 0;color:var(--text-2)">
             <div class="spinner"></div><div>Loading settings...</div>
           </div>`,
    footer: `<button class="btn ghost" data-cancel>Cancel</button>
             <button class="btn primary" id="save" disabled>Save settings</button>`,
    onMount(root) {
      root.querySelector('[data-cancel]').onclick = closeModal;
      fillSettings(root).catch((e) => {
        root.querySelector('#settings-body').innerHTML =
          `<div class="alert">Could not load settings: ${esc(e.message)}</div>`;
      });
    },
  });
}

async function fillSettings(root) {
  const body = root.querySelector('#settings-body');
  const saveBtn = root.querySelector('#save');

  // settings first (instant), then the slower lists
  const cfg = await api('/api/settings');
  const [models] = await Promise.all([
    api('/api/models').catch(() => ({ groq: {} })),
    loadVoices(),
  ]);

  const chat = models.groq?.chat || [];
  const asr = models.groq?.asr || [];
  const store = cfg.key_storage || {};
  const settingsTuner = voiceTuner({
    value: { pitch: cfg.pitch ?? 0, speed: cfg.speed ?? 0,
             volume: cfg.volume ?? 0, soften: cfg.soften ?? 0 },
    getVoice: () => document.querySelector('#voice').value,
  });
  const opt = (list, sel) => list.map((m) =>
    `<option value="${esc(m)}"${m === sel ? ' selected' : ''}>${esc(m)}</option>`).join('');

  body.innerHTML = `
      <div class="field">
        <label for="key">Groq API key</label>
        <div style="display:flex;gap:8px">
          <input type="password" id="key" style="flex:1"
                 placeholder="${cfg.has_key ? esc(cfg.groq_api_key) : 'gsk_...'}">
          <button class="btn ghost" id="showkey" title="Show or hide">👁</button>
          <button class="btn ghost" id="testkey">Test key</button>
        </div>
        <div id="keyresult"></div>
        <div class="hint">Used for transcription (Whisper) and translation.
          Free key at <a href="https://console.groq.com/keys" target="_blank"
          rel="noopener" style="color:var(--accent)">console.groq.com/keys</a>.</div>
      </div>

      <div class="field">
        <label>Extra keys for translation</label>
        <div class="hint" style="margin:0 0 8px">Rate limits are metered
          <b>per account</b>, so a second key of your own shares the same budget
          and changes nothing. A key from a <b>different account</b> brings its
          own, and translation is split across all of them.</div>
        <div id="extrakeys">${
          (cfg.groq_api_keys || []).map((k, i) => `
            <div class="extrakey" style="display:flex;gap:8px;margin-bottom:7px">
              <input type="password" class="xkey" style="flex:1"
                     value="${esc(k)}" placeholder="gsk_...">
              <button class="btn ghost xtest" data-i="${i}">Test</button>
              <button class="btn ghost xdel" title="Remove">&#10005;</button>
            </div>`).join('')
        }</div>
        <button class="btn ghost" id="addkey" style="padding:6px 13px">+ Add a key</button>
        <div class="hint" id="keycount"></div>
        <div class="paths">
          <div><b>Saved to</b></div>
          <div>${store.settings_saved ? '&#10003;' : '&#183;'}
               <code>${esc(store.settings_file || '')}</code></div>
          <div>${store.cache_saved ? '&#10003;' : '&#183;'}
               <code>${esc(store.cache_file || '')}</code> <span>cached backup</span></div>
          ${store.from_env ? `<div>&#10003; <span>GROQ_API_KEY environment variable</span></div>` : ''}
          <div class="note">The cached copy restores your key automatically if
            settings.json is deleted. Both files stay on D: in this project folder.</div>
        </div>
      </div>

      <div class="row">
        <div class="field">
          <label for="asrm">Transcription model</label>
          ${asr.length ? `<select id="asrm">${opt(asr, cfg.asr_model)}</select>`
            : `<input type="text" id="asrm" value="${esc(cfg.asr_model)}">`}
          <div class="hint">large-v3 is the most accurate for Chinese; turbo is faster.</div>
        </div>
        <div class="field">
          <label for="prov">Translation runs on</label>
          <select id="prov">
            <option value="groq"${cfg.llm_provider === 'groq' ? ' selected' : ''}>Groq (recommended)</option>
            <option value="local"${cfg.llm_provider === 'local' ? ' selected' : ''}>Local server (LM Studio)</option>
          </select>
        </div>
      </div>

      <div class="field" id="groq-model-field">
        <label for="llmm">Translation model</label>
        ${chat.length ? `<select id="llmm">${opt(chat, cfg.llm_model)}</select>`
          : `<input type="text" id="llmm" value="${esc(cfg.llm_model)}">`}
        <div class="hint">${chat.length ? 'Live list from your Groq account.'
          : 'Add your API key and reopen Settings to load the live model list.'}</div>
      </div>

      <div class="field" id="helpers-field">
        <label>Also translate with these, in parallel</label>
        <div class="hint" style="margin:0 0 8px">Groq meters tokens per minute
          <b>per model</b>, so each extra model adds its own budget on the same
          API key. Two models is roughly twice the translation speed.</div>
        <div id="helpers">${
          (models.groq?.helpers || chat).filter((m) => m !== cfg.llm_model).map((m) => `
            <div class="check" style="margin-bottom:8px">
              <input type="checkbox" class="helper" value="${esc(m)}"
                     id="h_${esc(m).replace(/[^a-z0-9]/gi, '_')}"
                     ${(cfg.llm_helpers || []).includes(m) ? 'checked' : ''}>
              <label for="h_${esc(m).replace(/[^a-z0-9]/gi, '_')}">${esc(m)}</label>
            </div>`).join('') || '<div class="hint">No other chat models available.</div>'
        }</div>
        <div class="hint" id="tputnote"></div>
      </div>

      <div class="field hidden" id="local-model-field">
        <label for="localm">Local model</label>
        <input type="text" id="localm" value="${esc(cfg.local_model || '')}"
               placeholder="start LM Studio's server first">
        <div class="hint">Endpoint: ${esc(cfg.local_base_url)}</div>
      </div>

      <div class="field">
        <label for="voice">Default dub voice</label>
        <div style="display:flex;gap:8px">
          <select id="voice" style="flex:1">${voiceOptions(cfg.voice)}</select>
          <button class="btn ghost" id="prev">Preview</button>
        </div>
      </div>

      ${settingsTuner.html}

      <div class="field">
        <label for="speed">Maximum speed-up when a line runs long
          <b id="speedval" style="color:var(--accent)">${cfg.max_speedup}x</b></label>
        <input type="range" id="speed" min="1.1" max="2.2" step="0.05"
               value="${cfg.max_speedup}" style="width:100%">
        <div class="hint">English is usually wordier than Chinese. When a translated
          line will not fit its slot it is spoken faster, up to this limit, so the
          audio stays locked to the picture.</div>
      </div>

      <div class="check">
        <input type="checkbox" id="keeporig" ${cfg.keep_original_audio ? 'checked' : ''}>
        <label for="keeporig">Keep the original audio quietly underneath
          <div class="hint">Preserves music and ambience, but you will hear both languages
            at once on a dialogue-heavy video. Off by default.</div></label>
      </div>

      <div class="field">
        <label for="gain">Original audio level <b id="gainval" style="color:var(--accent)">${Math.round(cfg.original_audio_gain * 100)}%</b></label>
        <input type="range" id="gain" min="0" max="0.5" step="0.01"
               value="${cfg.original_audio_gain}" style="width:100%">
      </div>

      <div class="check">
        <input type="checkbox" id="burn" ${cfg.burn_subtitles ? 'checked' : ''}>
        <label for="burn">Burn English subtitles into the picture
          <div class="hint">Permanently draws subtitles on the video. Slower, since the
            video must be re-encoded. Subtitles are available as a toggle either way.</div></label>
      </div>

      <h4>Downloads</h4>
      <div class="field">
        <label for="ytbrowser">Sign in to YouTube as</label>
        <select id="ytbrowser">
          <option value=""${cfg.ytdlp_browser ? '' : ' selected'}>Don't use a browser's cookies</option>
          ${['chrome', 'brave', 'edge', 'firefox', 'opera', 'vivaldi', 'chromium']
            .map((b) => `<option value="${b}"${cfg.ytdlp_browser === b ? ' selected' : ''}>${
              b[0].toUpperCase() + b.slice(1)}</option>`).join('')}
        </select>
        <div class="hint">YouTube now asks many downloads to prove they are not a bot.
          Picking a browser you are signed in to YouTube with gets past it.
          <b>That browser must be closed</b> when you download - its cookies cannot
          be read while it is running.</div>
      </div>`;

  saveBtn.disabled = false;

  const getTune = settingsTuner.mount(root);
  const helpers = () => [...root.querySelectorAll('.helper:checked')].map((c) => c.value);
  // Measured rates, not the token budget divided by a guess. A model that
  // cannot switch its reasoning off spends roughly twice as many tokens per
  // subtitle line, so it contributes about half as much as its budget suggests.
  const linesPerMin = (m) => (/gpt-oss|deepseek-r1|reason|think/i.test(m) ? 50 : 110);
  const tput = () => {
    const chosen = helpers();
    const n = chosen.length + 1;
    const primary = root.querySelector('#llmm')?.value || cfg.llm_model;
    const rate = linesPerMin(primary)
      + chosen.reduce((a, m) => a + linesPerMin(m), 0);
    const note = root.querySelector('#tputnote');
    if (note) {
      note.innerHTML = `Combined budget: <b style="color:var(--accent)">`
        + `${(n * 8000).toLocaleString()} tokens/min</b> across ${n} model`
        + `${n > 1 ? 's' : ''} - roughly <b style="color:var(--accent)">${rate}`
        + `</b> subtitle lines per minute.`
        + (n > 1 ? ' Names and genders are agreed up front so the models stay'
          + ' consistent. Reasoning models such as gpt-oss cost about twice the'
          + ' tokens per line, so two models is nearer 1.5x than 2x.' : '');
    }
  };
  root.querySelectorAll('.helper').forEach((c) => { c.onchange = tput; });
  root.querySelector('#llmm')?.addEventListener('change', tput);
  tput();

  /* ---- extra keys ------------------------------------------------------ */
  const extraBox = root.querySelector('#extrakeys');
  const extraKeys = () => [...root.querySelectorAll('.xkey')]
    .map((i) => i.value.trim()).filter(Boolean);

  const countKeys = () => {
    const n = extraKeys().length + 1;
    const note = root.querySelector('#keycount');
    if (!note) return;
    note.innerHTML = n === 1
      ? 'One key. Translation runs on its buckets alone.'
      : `<b style="color:var(--accent)">${n} keys</b> - translation is spread `
        + `across all of them. Only worth it if they are different accounts.`;
  };

  const wireRow = (row) => {
    row.querySelector('.xdel').onclick = (e) => {
      e.preventDefault(); row.remove(); countKeys();
    };
    row.querySelector('.xtest').onclick = async (e) => {
      e.preventDefault();
      const btn = e.target;
      const val = row.querySelector('.xkey').value.trim();
      if (!val) { toast('Paste a key first.', true); return; }
      const label = btn.textContent;
      btn.disabled = true; btn.textContent = '...';
      try {
        const r = await jsonPost('/api/settings/test-key', { key: val });
        toast(r.ok ? r.detail : r.detail, !r.ok);
      } catch (err) { toast(err.message, true); }
      btn.disabled = false; btn.textContent = label;
    };
  };
  extraBox.querySelectorAll('.extrakey').forEach(wireRow);

  root.querySelector('#addkey').onclick = (e) => {
    e.preventDefault();
    const row = document.createElement('div');
    row.className = 'extrakey';
    row.style.cssText = 'display:flex;gap:8px;margin-bottom:7px';
    row.innerHTML = `<input type="password" class="xkey" style="flex:1" placeholder="gsk_...">
      <button class="btn ghost xtest">Test</button>
      <button class="btn ghost xdel" title="Remove">&#10005;</button>`;
    extraBox.append(row);
    wireRow(row);
    row.querySelector('.xkey').oninput = countKeys;
    row.querySelector('.xkey').focus();
    countKeys();
  };
  extraBox.querySelectorAll('.xkey').forEach((i) => { i.oninput = countKeys; });
  countKeys();

  const keyInput = root.querySelector('#key');
  const keyResult = root.querySelector('#keyresult');
  const prov = root.querySelector('#prov');
  const speed = root.querySelector('#speed');
  const gain = root.querySelector('#gain');

  /* show / hide the key */
  root.querySelector('#showkey').onclick = (e) => {
    e.preventDefault();
    keyInput.type = keyInput.type === 'password' ? 'text' : 'password';
  };

  /* validate the key against Groq before committing to it */
  root.querySelector('#testkey').onclick = async (e) => {
    e.preventDefault();
    const btn = e.target;
    btn.disabled = true; btn.textContent = 'Testing...';
    keyResult.innerHTML = '';
    try {
      const r = await jsonPost('/api/settings/test-key',
        { groq_api_key: keyInput.value.trim(), save: true });
      keyResult.innerHTML = `<div class="alert ${r.ok ? 'info' : ''}"
        style="margin:10px 0 0">${esc(r.detail)}</div>`;
      if (r.ok) {
        keyInput.value = '';
        keyInput.placeholder = 'saved';
        toast('API key verified and saved');
        if (r.chat?.length) {
          const sel = root.querySelector('#llmm');
          if (sel?.tagName === 'SELECT') {
            sel.innerHTML = r.chat.map((m) =>
              `<option value="${esc(m)}"${m === cfg.llm_model ? ' selected' : ''}>${esc(m)}</option>`).join('');
          }
        }
      }
    } catch (err) {
      keyResult.innerHTML = `<div class="alert" style="margin:10px 0 0">${esc(err.message)}</div>`;
    }
    btn.disabled = false; btn.textContent = 'Test key';
  };

  /* provider switch - probe LM Studio only when it is actually selected */
  let localProbed = false;
  const toggleProv = async () => {
    const local = prov.value === 'local';
    root.querySelector('#local-model-field').classList.toggle('hidden', !local);
    root.querySelector('#groq-model-field').classList.toggle('hidden', local);
    root.querySelector('#helpers-field')?.classList.toggle('hidden', local);
    if (!local || localProbed) return;
    localProbed = true;
    const field = root.querySelector('#local-model-field');
    const hint = field.querySelector('.hint');
    hint.textContent = 'Checking for a local server...';
    const m = await api('/api/models?local=1').catch(() => ({ local: [] }));
    const sel = field.querySelector('#localm');
    if (m.local?.length) {
      sel.outerHTML = `<select id="localm">${m.local.map((x) =>
        `<option value="${esc(x)}"${x === cfg.local_model ? ' selected' : ''}>${esc(x)}</option>`).join('')}</select>`;
      hint.textContent = `${m.local.length} model(s) loaded at ${cfg.local_base_url}.`;
    } else {
      hint.textContent = `Nothing reachable at ${cfg.local_base_url}. `
        + 'Start LM Studio and enable its local server, then reopen Settings.';
    }
  };
  prov.onchange = toggleProv;
  toggleProv();

  speed.oninput = () => {
    root.querySelector('#speedval').textContent = (+speed.value).toFixed(2) + 'x';
  };
  gain.oninput = () => {
    root.querySelector('#gainval').textContent = Math.round(gain.value * 100) + '%';
  };

  root.querySelector('#prev').onclick = (e) => {
    e.preventDefault();
    previewVoice(root.querySelector('#voice').value, e.target, getTune());
  };

  saveBtn.onclick = async () => {
    saveBtn.disabled = true; saveBtn.textContent = 'Saving...';
    const payload = {
      asr_model: root.querySelector('#asrm').value,
      llm_provider: prov.value,
      llm_model: root.querySelector('#llmm').value,
      llm_helpers: helpers(),
      groq_api_keys: extraKeys(),
      local_model: root.querySelector('#localm')?.value || '',
      voice: root.querySelector('#voice').value,
      ...getTune(),
      max_speedup: parseFloat(speed.value),
      keep_original_audio: root.querySelector('#keeporig').checked,
      original_audio_gain: parseFloat(gain.value),
      burn_subtitles: root.querySelector('#burn').checked,
      ytdlp_browser: root.querySelector('#ytbrowser')?.value || '',
    };
    const key = keyInput.value.trim();
    if (key) payload.groq_api_key = key;
    try {
      const saved = await jsonPost('/api/settings', payload);
      closeModal();
      toast(saved.has_key ? 'Settings saved' : 'Settings saved - no API key yet');
    } catch (err) {
      toast(err.message, true);
      saveBtn.disabled = false; saveBtn.textContent = 'Save settings';
    }
  };
}

/* ---------------------------------------------------------------- home --- */
function statusChip(v) {
  if (v.status === 'ready') return '';
  const label = { processing: 'Processing', queued: 'Queued', failed: 'Failed',
                  downloading: 'Downloading', downloaded: 'Not dubbed' }[v.status] || v.status;
  return `<span class="chip ${esc(v.status)}">${label}</span>`;
}

function card(v) {
  const busy = v.status === 'processing' || v.status === 'queued'
            || v.status === 'downloading';
  const thumb = v.has_thumb
    ? `<img src="/media/${v.id}/thumb.jpg" alt="" loading="lazy">`
    : `<div style="color:var(--text-3);font-size:32px">🎬</div>`;

  const meta = v.status === 'ready'
    ? `${v.line_count} lines &middot; ${esc(v.voice ? v.voice.split('-').pop().replace('Neural', '') : '')} &middot; ${ago(v.created_at)}`
    : v.status === 'failed'
      ? `<span style="color:var(--red)">${esc((v.error || 'Failed').slice(0, 90))}</span>`
      : `${esc(v.stage || 'Waiting')}`;

  return `
    <div class="card${busy ? ' disabled' : ''}" data-id="${v.id}" data-status="${v.status}">
      <div class="thumb">
        ${thumb}
        ${statusChip(v)}
        ${v.duration ? `<span class="badge">${hhmmss(v.duration)}</span>` : ''}
        ${watchedBar(v)}
      </div>
      <div class="card-body">
        <div style="flex:1;min-width:0">
          <div class="card-title">${esc(v.title)}</div>
          <div class="card-meta">${meta}</div>
        </div>
        <button class="btn icon ghost" data-menu="${v.id}" title="Options">&#8942;</button>
      </div>
      ${v.status === 'downloaded' ? `<div class="card-act">
          <button class="btn primary" data-translate="${v.id}">Translate</button>
        </div>` : ''}
      ${busy ? `<div class="progress-wrap">
          <div class="bar${v.progress ? '' : ' indeterminate'}"><i style="width:${v.progress || 0}%"></i></div>
          <div class="stage">${esc(v.stage || '')}${v.progress ? ` &middot; ${v.progress}%` : ''}</div>
          <div class="live-row">${liveButton(v)}</div>
        </div>` : ''}
    </div>`;
}

/* Badges on a transcript line saying what the pipeline had to do to it. The
   point is to make the compromises findable: a line that was cut short is one
   the viewer can shorten by hand, but only if they can see which one it was. */
function lineMarks(s) {
  const out = [];
  if (s.dub_speedup > 1.05) out.push(`<span class="fast">${(+s.dub_speedup).toFixed(2)}x</span>`);
  if (s.fixed) out.push('<span class="mark fixed" title="Rewritten by the repair pass">rewritten</span>');
  if (s.clipped) out.push('<span class="mark cut" title="Too long for its slot - the end was cut">cut short</span>');
  return out.join('');
}

/* How far into an episode the viewer got, drawn across the bottom of the
   thumbnail the way every video service does it. At an hour a title, "did I
   already watch this one, and where did I stop" is a real question. */
function watchedBar(v) {
  const at = +v.position || 0;
  if (!(at > 20) || !(v.duration > 0)) return '';
  const pct = Math.min(100, (at / v.duration) * 100);
  if (pct > 97) return '<span class="seen">Watched</span>';
  return `<div class="watched" title="Stopped at ${hhmmss(at)}">
            <i style="width:${pct.toFixed(1)}%"></i></div>`;
}

/* Watching starts as soon as the first window of dub is published, so a long
   video does not have to be finished before it can be played. */
function liveButton(v) {
  const s = (v.stream && v.stream.seconds) || 0;
  if (s < 5) return '';
  return `<a class="btn live" href="#/watch/${v.id}">
      <span class="dot"></span>Watch now &middot; ${hhmmss(s)} ready</a>`;
}

/* A downloaded video has not been dubbed and is not on its way to being dubbed.
   Fetching a link and spending an hour of Groq budget on it are separate
   decisions, so they are separate shelves. */
const DOWNLOAD_STATES = ['downloading', 'downloaded'];

async function renderHome(section = 'dubbed') {
  let videos = [];
  try { videos = await api('/api/videos'); } catch (e) { toast(e.message, true); }

  const mine = videos.filter((v) => DOWNLOAD_STATES.includes(v.status)
    === (section === 'downloads'));
  const q = SEARCH.toLowerCase();
  const shown = q ? mine.filter((v) => v.title.toLowerCase().includes(q)) : mine;
  const waiting = videos.filter((v) => DOWNLOAD_STATES.includes(v.status)).length;

  const tabs = `
    <div class="tabs">
      <a href="#/" class="${section === 'dubbed' ? 'on' : ''}">Dubbed</a>
      <a href="#/downloads" class="${section === 'downloads' ? 'on' : ''}">Downloads${
        waiting ? `<span class="pill">${waiting}</span>` : ''}</a>
    </div>`;

  const empty = section === 'downloads'
    ? `<div class="empty">
         <h3>No downloads yet</h3>
         <p>Paste a video link and pick a quality. It is saved here without
            being dubbed,<br>so you choose when to spend the translation on it.</p>
         <button class="btn primary" onclick="window.__download()" style="margin-top:14px">Paste a link</button>
       </div>`
    : `<div class="empty">
         <h3>Nothing dubbed yet</h3>
         <p>Upload a video or paste a link, then press Translate.<br>
            It is transcribed, translated to English, re-voiced and re-synced.</p>
         <button class="btn primary" onclick="window.__upload()" style="margin-top:14px">Upload a video</button>
       </div>`;

  view.innerHTML = `
    ${tabs}
    <div class="section-head">
      <h2>${q ? 'Search results' : section === 'downloads' ? 'Downloaded' : 'Your library'}</h2>
      <span class="count">${shown.length} video${shown.length === 1 ? '' : 's'}</span>
    </div>
    ${shown.length ? `<div class="grid">${shown.map(card).join('')}</div>`
      : (q ? `<div class="empty"><h3>Nothing matches "${esc(SEARCH)}"</h3></div>` : empty)}`;

  view.querySelectorAll('.card').forEach((el) => {
    el.onclick = (e) => {
      if (e.target.closest('[data-menu]') || e.target.closest('[data-translate]')) return;
      const { id, status } = el.dataset;
      if (status === 'ready') location.hash = `#/watch/${id}`;
      else if (status === 'failed') showFailure(id);
      else if (status === 'downloaded') toast('Press Translate to dub this one.');
      else toast('Still working - this card updates on its own.');
    };
  });
  view.querySelectorAll('[data-menu]').forEach((btn) => {
    btn.onclick = (e) => { e.stopPropagation(); cardMenu(btn.dataset.menu); };
  });
  view.querySelectorAll('[data-translate]').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      btn.disabled = true;
      btn.textContent = 'Queued';
      try {
        await api(`/api/videos/${btn.dataset.translate}/translate`, { method: 'POST' });
        toast('Queued for dubbing - it moves to Dubbed as it starts');
        location.hash = '#/';
        render();
      } catch (err) {
        toast(err.message, true);
        btn.disabled = false;
        btn.textContent = 'Translate';
      }
    };
  });

  return videos;
}

async function cardMenu(id) {
  const v = await api(`/api/videos/${id}`).catch(() => null);
  if (!v) return;
  const busy = v.status === 'processing' || v.status === 'queued';
  modal({
    title: v.title,
    body: `
      <div class="statgrid">
        <div><span>Status</span><b>${esc(v.status)}</b></div>
        <div><span>Length</span><b>${hhmmss(v.duration)}</b></div>
        <div><span>Lines</span><b>${v.line_count || 0}</b></div>
        <div><span>Added</span><b>${ago(v.created_at)}</b></div>
      </div>
      ${v.error ? `<div class="alert">${esc(v.error)}</div>` : ''}
      <div style="display:flex;flex-wrap:wrap;gap:8px">
        ${v.status === 'ready' ? `<a class="btn ghost" href="#/watch/${v.id}" onclick="window.__close()">Open</a>
          <a class="btn ghost" href="/media/${v.id}/video" download="${esc(v.title)}.mp4">Download video</a>
          <a class="btn ghost" href="/media/${v.id}/english.srt" download>English .srt</a>
          <a class="btn ghost" href="/media/${v.id}/original.srt" download>Original .srt</a>` : ''}
        ${v.status === 'failed' ? `<button class="btn ghost" id="retry">Try again</button>` : ''}
        ${v.status === 'ready' && !busy ? `<button class="btn ghost" id="retrans">Translate again</button>` : ''}
      </div>
      ${v.status === 'ready' ? `<div class="hint" style="margin-top:10px">Translate
        again re-runs the translation on the transcript you already have, then
        re-voices it. It does not re-transcribe, so it costs no Whisper time -
        use it after changing the translation model or adding parallel models.</div>` : ''}`,
    footer: `<button class="btn danger" id="del" ${busy ? 'disabled' : ''}>Delete</button>
             <button class="btn ghost" data-cancel>Close</button>`,
    onMount(root) {
      root.querySelector('[data-cancel]').onclick = closeModal;
      const retry = root.querySelector('#retry');
      if (retry) retry.onclick = async () => {
        try { await jsonPost(`/api/videos/${id}/retry`, {}); closeModal(); toast('Resuming where it stopped'); render(); }
        catch (e) { toast(e.message, true); }
      };
      const retrans = root.querySelector('#retrans');
      if (retrans) retrans.onclick = async () => {
        if (!confirm('Re-translate this video from its existing transcript?\n\n'
          + 'The current English text and dubbed audio are replaced.')) return;
        try { await jsonPost(`/api/videos/${id}/retranslate`, {}); closeModal(); toast('Re-translating'); render(); }
        catch (e) { toast(e.message, true); }
      };
      root.querySelector('#del').onclick = async () => {
        if (!confirm(`Delete "${v.title}" and all of its files?`)) return;
        try { await api(`/api/videos/${id}`, { method: 'DELETE' }); closeModal(); toast('Deleted'); render(); }
        catch (e) { toast(e.message, true); }
      };
    },
  });
}

async function showFailure(id) {
  const v = await api(`/api/videos/${id}`).catch(() => null);
  if (!v) return;
  modal({
    title: 'Processing failed',
    body: `<div class="alert">${esc(v.error || 'Unknown error')}</div>
      <p style="color:var(--text-2)">Common causes: no Groq API key, an expired model id,
      a rate limit, or a video with no speech in it.</p>
      <p style="color:var(--text-2)">Trying again picks up from the last stage that finished -
      transcription, translation and speech already done are not repeated.</p>`,
    footer: `<button class="btn ghost" data-cancel>Close</button>
             <button class="btn primary" id="retry">Try again</button>`,
    onMount(root) {
      root.querySelector('[data-cancel]').onclick = closeModal;
      root.querySelector('#retry').onclick = async () => {
        try { await jsonPost(`/api/videos/${id}/retry`, {}); closeModal(); toast('Resuming where it stopped'); render(); }
        catch (e) { toast(e.message, true); }
      };
    },
  });
}

/* --------------------------------------------------------------- watch --- */
async function renderWatch(id) {
  let v;
  try { v = await api(`/api/videos/${id}`); }
  catch (e) { view.innerHTML = `<div class="empty"><h3>${esc(e.message)}</h3>
    <a class="btn ghost" href="#/">Back to library</a></div>`; return; }

  // A video still being dubbed is watchable as soon as it has published a
  // stream; only one with nothing to play yet goes back to the library.
  const live = v.status !== 'ready';
  if (live && !((v.stream && v.stream.seconds) > 0)) { location.hash = '#/'; return; }

  const segs = v.segments || [];
  const st = v.stats || {};

  view.innerHTML = `
    <div class="watch">
      <div>
        <div class="player-wrap">
          <video id="player" controls preload="metadata"
                 ${live ? '' : `src="/media/${id}/video?t=${v.updated_at}"`}>
            ${live ? '' : `<track id="tr-en" kind="subtitles" srclang="en" label="English"
                   src="/media/${id}/english.vtt?t=${v.updated_at}" default>
            <track id="tr-src" kind="subtitles" srclang="${esc(v.source_lang || 'zh')}" label="Original"
                   src="/media/${id}/original.vtt?t=${v.updated_at}">`}
          </video>
        </div>
        ${live ? `<div class="livebar" id="livebar">
            <span class="dot"></span>
            <b id="live-stage">${esc(v.stage || 'Dubbing')}</b>
            <span class="live-ready" id="live-ready"></span>
            <div class="live-track"><i id="live-fill"></i></div>
          </div>` : ''}

        <h1>${esc(v.title)}</h1>

        <div class="toolbar">
          ${live ? '' : `<div class="pill-group" id="audio-toggle">
            <button data-src="0" class="on">English dub</button>
            <button data-src="1">Original audio</button>
          </div>
          <div class="pill-group" id="sub-toggle">
            <button data-sub="en" class="on">English subs</button>
            <button data-sub="src">Original subs</button>
            <button data-sub="off">Off</button>
          </div>`}
          <div class="speed">
            <label for="speed">Speed</label>
            <input id="speed" type="range" min="0.25" max="3" step="0.01" value="1"
                   aria-label="Playback speed">
            <button class="speed-val" id="speed-val"
                    title="Click to go back to normal speed">1.00&times;</button>
          </div>
          <span class="sep"></span>
          ${live ? '' : `<button class="btn ghost" id="btn-revoice">Voice &amp; audio</button>
          <a class="btn ghost" href="/media/${id}/video" download="${esc(v.title)}.mp4">Download</a>
          <button class="btn icon ghost" id="btn-more" title="Options">&#8942;</button>`}
        </div>

        <div class="statgrid">
          <div><span>Length</span><b>${hhmmss(v.duration)}</b></div>
          <div><span>Lines dubbed</span><b>${st.lines ?? segs.length}</b></div>
          <div><span>Voice</span><b>${esc((v.voice || '').split('-').pop().replace('Neural', '') || '-')}</b></div>
          <div><span>Sped up to fit</span><b>${st.compressed ?? 0} lines</b></div>
          <div><span>Peak rate</span><b>${st.max_speedup_used ? st.max_speedup_used + 'x' : '1x'}</b></div>
          <div><span>Max drift</span><b>${(+(st.drift ?? 0)).toFixed(2)}s</b></div>
          ${st.fixed ? `<div><span>Lines rewritten</span><b>${st.fixed}</b></div>` : ''}
          ${st.clipped ? `<div><span>Trimmed to fit</span><b>${st.clipped} lines</b></div>` : ''}
        </div>
        ${st.clipped ? `<div class="note">${st.clipped} line(s) were too long to
          say in their slot and had to be cut short. They are marked in the
          transcript - shorten one and re-dub if it matters.</div>` : ''}
        ${st.failed ? `<div class="alert">${st.failed} line(s) could not be voiced and are silent.</div>` : ''}
      </div>

      <aside class="panel">
        <div class="panel-head">
          <h3>Transcript</h3>
          <button class="btn ghost" id="btn-edit" style="padding:5px 12px;font-size:13px">Edit</button>
        </div>
        <div class="panel-body" id="lines">
          ${segs.map((s) => `
            <div class="line${s.clipped ? ' flagged' : ''}" data-id="${s.id}"
                 data-start="${s.start}" data-end="${s.end}">
              <time>${hhmmss(s.start)}</time>
              <div>
                <div class="en" data-id="${s.id}">${esc(s.en || '')}</div>
                <div class="src">${esc(s.text || '')}${lineMarks(s)}</div>
              </div>
            </div>`).join('')}
        </div>
      </aside>
    </div>`;

  const player = document.getElementById('player');
  const linesBox = document.getElementById('lines');

  /* Playback speed. A dub is worth slowing down or nudging along by a few
     percent rather than the player's fixed 0.25 jumps, so this is a continuous
     control: 0.01 steps, and the arrow keys move it one step at a time once it
     has focus. The reading doubles as the reset - click it to return to 1x,
     which is otherwise fiddly to land on by dragging. */
  const speed = document.getElementById('speed');
  const speedVal = document.getElementById('speed-val');
  const setSpeed = (raw, save = true) => {
    const rate = Math.min(3, Math.max(0.25, Math.round((+raw || 1) * 100) / 100));
    player.playbackRate = rate;
    // Keep the voice at its own pitch instead of chipmunking it. Browsers
    // default to this, but the prefixed forms are cheap insurance.
    player.preservesPitch = true;
    player.mozPreservesPitch = player.webkitPreservesPitch = true;
    speed.value = rate;
    speedVal.textContent = rate.toFixed(2) + '×';
    speedVal.classList.toggle('changed', Math.abs(rate - 1) > 0.004);
    if (save) { try { localStorage.setItem('dubline.speed', rate); } catch (e) { /* private mode */ } }
  };
  speed.oninput = () => setSpeed(speed.value);
  speedVal.onclick = () => setSpeed(1);
  // Chosen once and kept, the way every other player behaves - and the reading
  // sits next to the slider so a rate carried over from last time is visible
  // rather than a mystery.
  let saved = 1;
  try { saved = parseFloat(localStorage.getItem('dubline.speed')) || 1; } catch (e) { /* ignore */ }
  setSpeed(saved, false);
  // The element re-applies its default rate when a new source is attached, so
  // reassert after the picture is actually there - live streams attach late.
  player.addEventListener('loadedmetadata', () => setSpeed(speed.value, false));

  /* Pick up where you stopped. An episode here runs an hour, so starting from
     zero every time is the difference between finishing one and not. Ignored
     near the very start and the very end, where resuming is just annoying, and
     for a live stream, which has its own idea of where to begin. */
  let lastSaved = 0;
  const saveAt = (at, force) => {
    if (live || !isFinite(at)) return;
    if (!force && Math.abs(at - lastSaved) < 5) return;
    lastSaved = at;
    fetch(`/api/videos/${id}/position`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ at }), keepalive: true,
    }).catch(() => {});
  };
  const resumeAt = +v.position || 0;
  if (!live && resumeAt > 20 && (!v.duration || resumeAt < v.duration - 20)) {
    player.addEventListener('loadedmetadata', () => {
      if (!RESUME) player.currentTime = resumeAt;
      toast(`Resuming from ${hhmmss(resumeAt)}`);
    }, { once: true });
  }
  player.addEventListener('timeupdate', () => saveAt(player.currentTime));
  player.addEventListener('pause', () => saveAt(player.currentTime, true));
  const onLeave = () => saveAt(player.currentTime, true);
  window.addEventListener('pagehide', onLeave);
  KEYS.cleanup.push(() => window.removeEventListener('pagehide', onLeave));

  /* Keyboard control. Everything here is what a video player is expected to do
     and what this one could not: without it there was no way to step back four
     seconds to catch a line, which is the single most common thing to want
     while checking a dub. */
  const nudge = (d) => {
    player.currentTime = Math.max(0, Math.min(player.duration || 1e9,
                                              player.currentTime + d));
    toast(`${d > 0 ? '+' : ''}${d}s`);
  };
  const bumpSpeed = (d) => setSpeed(Math.round((player.playbackRate + d) * 100) / 100);
  const onKey = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const el = document.activeElement;
    // Never steal a key from something the viewer is typing into.
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
               || el.isContentEditable)) return;
    const k = e.key;
    const map = {
      ' ': () => (player.paused ? player.play() : player.pause()),
      k: () => (player.paused ? player.play() : player.pause()),
      ArrowLeft: () => nudge(-5), ArrowRight: () => nudge(5),
      j: () => nudge(-10), l: () => nudge(10),
      ArrowUp: () => { player.volume = Math.min(1, player.volume + 0.1); },
      ArrowDown: () => { player.volume = Math.max(0, player.volume - 0.1); },
      m: () => { player.muted = !player.muted; },
      f: () => (document.fullscreenElement ? document.exitFullscreen()
                                           : player.requestFullscreen?.()),
      '[': () => bumpSpeed(-0.05), ']': () => bumpSpeed(0.05),
      '\\': () => setSpeed(1),
      c: () => view.querySelector('#sub-toggle button.on')
        ?.nextElementSibling?.click() || view.querySelector('#sub-toggle button')?.click(),
    };
    const act = map[k] || map[k.toLowerCase?.()];
    if (!act) return;
    e.preventDefault();
    act();
  };
  document.addEventListener('keydown', onKey);
  KEYS.cleanup.push(() => document.removeEventListener('keydown', onKey));

  if (live) startLive(id, player, v);

  // Coming back from the live view when the dub finished: carry the viewer's
  // position across so the switch to the finished file is not a restart.
  if (!live && RESUME && RESUME.id === id) {
    const { at, playing } = RESUME;
    RESUME = null;
    player.addEventListener('loadedmetadata', () => {
      player.currentTime = at;
      if (playing) player.play().catch(() => {});
    }, { once: true });
  }

  /* subtitle toggle */
  const setSubs = (which) => {
    const tracks = player.textTracks;
    for (let i = 0; i < tracks.length; i++) {
      const want = (which === 'en' && tracks[i].label === 'English')
        || (which === 'src' && tracks[i].label === 'Original');
      tracks[i].mode = want ? 'showing' : 'disabled';
    }
  };
  view.querySelectorAll('#sub-toggle button').forEach((b) => {
    b.onclick = () => {
      view.querySelectorAll('#sub-toggle button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      setSubs(b.dataset.sub);
    };
  });
  if (!live) player.addEventListener('loadedmetadata', () => setSubs(
    view.querySelector('#sub-toggle .on').dataset.sub), { once: true });

  /* audio source toggle - keeps position and play state */
  view.querySelectorAll('#audio-toggle button').forEach((b) => {
    b.onclick = () => {
      if (b.classList.contains('on')) return;
      view.querySelectorAll('#audio-toggle button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      const at = player.currentTime;
      const playing = !player.paused;
      const active = view.querySelector('#sub-toggle .on').dataset.sub;
      player.src = `/media/${id}/video?src=${b.dataset.src}&t=${v.updated_at}`;
      player.addEventListener('loadedmetadata', () => {
        player.currentTime = at;
        setSubs(active);
        if (playing) player.play();
      }, { once: true });
    };
  });

  /* click a line to jump there */
  linesBox.querySelectorAll('.line').forEach((el) => {
    el.onclick = (e) => {
      if (e.target.isContentEditable) return;
      player.currentTime = parseFloat(el.dataset.start);
      player.play();
    };
  });

  /* follow along while playing */
  let current = null;
  player.addEventListener('timeupdate', () => {
    const t = player.currentTime;
    const hit = segs.find((s) => t >= s.start && t < s.end);
    const nextId = hit ? hit.id : null;
    if (nextId === current) return;
    current = nextId;
    linesBox.querySelectorAll('.line.active').forEach((x) => x.classList.remove('active'));
    if (nextId === null) return;
    const el = linesBox.querySelector(`.line[data-id="${nextId}"]`);
    if (el && !linesBox.dataset.editing) {
      el.classList.add('active');
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  });

  /* transcript editing */
  const editBtn = document.getElementById('btn-edit');
  editBtn.onclick = async () => {
    const editing = linesBox.dataset.editing === '1';
    if (!editing) {
      linesBox.dataset.editing = '1';
      editBtn.textContent = 'Save';
      editBtn.classList.add('primary');
      linesBox.querySelectorAll('.en').forEach((el) => {
        el.contentEditable = 'true';
        el.dataset.orig = el.textContent.trim();
      });
      toast('Edit the English lines, then Save');
      return;
    }
    const edits = {};
    linesBox.querySelectorAll('.en').forEach((el) => {
      const now = el.textContent.trim();
      if (now !== el.dataset.orig) edits[el.dataset.id] = now;
      el.contentEditable = 'false';
    });
    delete linesBox.dataset.editing;
    editBtn.textContent = 'Edit';
    editBtn.classList.remove('primary');

    const n = Object.keys(edits).length;
    if (!n) { toast('No changes'); return; }
    try {
      await jsonPost(`/api/videos/${id}/segments`, { edits }, 'PUT');
      if (confirm(`Saved ${n} edited line(s). Re-generate the dubbed audio now?\n\nThis re-voices the whole video (a few minutes).`)) {
        await jsonPost(`/api/videos/${id}/revoice`, {});
        toast('Re-dubbing started');
        location.hash = '#/';
        render();
      } else {
        toast('Subtitles updated. The audio still uses the old wording.');
      }
    } catch (e) { toast(e.message, true); }
  };

  /* voice change */
  const revoiceBtn = document.getElementById('btn-revoice');
  if (revoiceBtn) revoiceBtn.onclick = async () => {
    await loadVoices();
    const cfg = await api('/api/settings').catch(() => ({}));
    const bgOn = v.stats?.keep_original_audio ?? cfg.keep_original_audio ?? false;
    const curTune = {
      pitch: v.stats?.pitch ?? cfg.pitch ?? 0,
      speed: v.stats?.speed ?? cfg.speed ?? 0,
      volume: v.stats?.volume ?? cfg.volume ?? 0,
      soften: v.stats?.soften ?? cfg.soften ?? 0,
    };
    const tuner = voiceTuner({
      value: curTune,
      getVoice: () => document.querySelector('#nv').value,
      sample: (v.segments || []).find((s) => (s.en || '').length > 40)?.en
        || 'I never lost. You only thought you had won.',
    });

    modal({
      title: 'Voice and audio mix',
      body: `<div class="field">
          <label for="nv">Voice</label>
          <div style="display:flex;gap:8px">
            <select id="nv" style="flex:1">${voiceOptions(v.voice)}</select>
            <button class="btn ghost" id="prev">Preview</button>
          </div>
          <div class="hint">The existing translation is reused, so changing the
            voice only re-runs speech synthesis - no API calls.</div>
        </div>
        ${tuner.html}
        <div class="check">
          <input type="checkbox" id="bg" ${bgOn ? 'checked' : ''}>
          <label for="bg">Keep the original audio quietly underneath
            <div class="hint">Preserves music and background sound, but on a
              dialogue-heavy video you hear both languages at once. Leave off
              unless the video is mostly music.</div></label>
        </div>
        <div class="field ${bgOn ? '' : 'hidden'}" id="gainfield">
          <label for="bgain">Original audio level
            <b id="bgv" style="color:var(--accent)">${Math.round((cfg.original_audio_gain ?? 0.06) * 100)}%</b></label>
          <input type="range" id="bgain" min="0.01" max="0.30" step="0.01"
                 value="${cfg.original_audio_gain ?? 0.06}" style="width:100%">
        </div>`,
      footer: `<button class="btn ghost" data-cancel>Cancel</button>
               <button class="btn primary" id="go">Apply</button>`,
      onMount(root) {
        const bg = root.querySelector('#bg');
        const gain = root.querySelector('#bgain');
        const getTune = tuner.mount(root);
        bg.onchange = () => root.querySelector('#gainfield').classList.toggle('hidden', !bg.checked);
        gain.oninput = () => {
          root.querySelector('#bgv').textContent = Math.round(gain.value * 100) + '%';
        };
        root.querySelector('[data-cancel]').onclick = closeModal;
        root.querySelector('#prev').onclick = (e) => {
          e.preventDefault();
          previewVoice(root.querySelector('#nv').value, e.target, getTune());
        };
        root.querySelector('#go').onclick = async () => {
          const voice = root.querySelector('#nv').value;
          const tune = getTune();
          const body = {
            keep_original_audio: bg.checked,
            original_audio_gain: parseFloat(gain.value),
          };
          // nothing about the voice changed - re-muxing reuses the rendered
          // dub track and takes seconds instead of re-synthesising every line
          const mixOnly = voice === v.voice
            && Object.keys(tune).every((k) => tune[k] === curTune[k]);
          try {
            if (!mixOnly) Object.assign(body, { voice }, tune);
            await jsonPost(`/api/videos/${id}/${mixOnly ? 'remix' : 'revoice'}`, body);
            closeModal();
            toast(mixOnly ? 'Re-mixing audio' : 'Re-dubbing started');
            location.hash = '#/';
            render();
          } catch (e) { toast(e.message, true); }
        };
      },
    });
  };

  const moreBtn = document.getElementById('btn-more');
  if (moreBtn) moreBtn.onclick = () => cardMenu(id);
}

/* ---------------------------------------------------------- live watch --- */
/* The dub is served as HLS while it is still being made. The picture is whole
   from the start, so seeking anywhere works; the audio playlist grows, and
   hls.js picks up new segments each time it reloads the playlist. */
function startLive(id, player, v) {
  const url = `/api/videos/${id}/stream/master.m3u8`;
  FINISHED = false;

  if (window.Hls && Hls.isSupported()) {
    const hls = new Hls({
      // The playlist is still being written, so a 404 on a segment the player
      // guessed at is normal rather than fatal - keep retrying instead.
      manifestLoadingMaxRetry: 20,
      levelLoadingMaxRetry: 20,
      fragLoadingMaxRetry: 20,
      // Both playlists are open-ended while dubbing, which makes this look
      // like a live stream. It is not: the viewer wants the beginning, not
      // whatever was encoded a moment ago.
      startPosition: 0,
      liveDurationInfinity: false,
      lowLatencyMode: false,
    });
    hls.loadSource(url);
    hls.attachMedia(player);
    hls.on(Hls.Events.ERROR, (_e, data) => {
      if (!data.fatal) return;
      // The stream is torn down the moment the dub finishes, so a fatal
      // network error here usually just means the finished file is ready.
      if (FINISHED) { render(); return; }
      if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad();
      else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
    });
    player.__hls = hls;
  } else {
    player.src = url;      // Safari plays HLS natively
  }

  const fill = document.getElementById('live-fill');
  const readyEl = document.getElementById('live-ready');
  const stageEl = document.getElementById('live-stage');

  const tick = async () => {
    const s = await api(`/api/videos/${id}/status`).catch(() => null);
    if (!s) return;
    const st = s.stream || {};
    const total = st.duration || v.duration || 0;
    const done = st.seconds || 0;
    if (fill && total) fill.style.width = Math.min(100, (100 * done) / total) + '%';
    if (readyEl) readyEl.textContent = `${hhmmss(done)} of ${hhmmss(total)} dubbed`;
    if (stageEl) stageEl.textContent = s.stage || 'Dubbing';

    if (s.status === 'ready' || s.status === 'failed') {
      FINISHED = true;
      clearInterval(liveTimer); liveTimer = null;
      // Reload into the finished page, keeping the viewer where they were.
      const at = player.currentTime, playing = !player.paused;
      RESUME = { id, at, playing };
      render();
    }
  };
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = setInterval(tick, 2000);
  tick();
}

let liveTimer = null;
let RESUME = null;
let FINISHED = false;
/* Listeners the watch page hangs on document and window. They have to be
   taken down when leaving the page: this is a single page app, so every
   visit to a video would otherwise leave another keyboard handler behind,
   and one press would seek by five seconds per video ever opened. */
const KEYS = { cleanup: [] };

/* -------------------------------------------------------------- router --- */
async function render() {
  if (poller) { clearInterval(poller); poller = null; }
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  KEYS.cleanup.splice(0).forEach((fn) => { try { fn(); } catch (e) {} });
  const old = document.getElementById('player');
  if (old && old.__hls) { try { old.__hls.destroy(); } catch (e) {} }
  const hash = location.hash || '#/';
  const watch = hash.match(/^#\/watch\/([a-z0-9]+)/i);

  if (watch) { await renderWatch(watch[1]); return; }

  const videos = await renderHome(hash.startsWith('#/downloads') ? 'downloads' : 'dubbed');
  // Downloads report progress the same way a dub does, so the same poller
  // keeps both shelves live.
  if (videos.some((v) => ['processing', 'queued', 'downloading'].includes(v.status))) {
    poller = setInterval(pollProgress, 1500);
  }
}

async function pollProgress() {
  const cards = [...document.querySelectorAll('.card[data-status="processing"], '
    + '.card[data-status="queued"], .card[data-status="downloading"]')];
  if (!cards.length) { clearInterval(poller); poller = null; return; }

  let finished = false;
  await Promise.all(cards.map(async (el) => {
    const s = await api(`/api/videos/${el.dataset.id}/status`).catch(() => null);
    if (!s) return;
    if (s.status !== el.dataset.status
        && ['ready', 'failed', 'downloaded'].includes(s.status)) {
      finished = true;
      return;
    }
    const bar = el.querySelector('.bar > i');
    const stage = el.querySelector('.stage');
    const meta = el.querySelector('.card-meta');
    const live = el.querySelector('.live-row');
    if (bar) {
      bar.style.width = (s.progress || 0) + '%';
      el.querySelector('.bar').classList.toggle('indeterminate', !s.progress);
    }
    if (stage) stage.innerHTML = esc(s.stage || '') + (s.progress ? ` &middot; ${s.progress}%` : '');
    if (meta) meta.innerHTML = esc(s.stage || 'Waiting');
    // The Watch button appears partway through the run, so it has to be drawn
    // here and not only by a full render - a card painted before the first
    // window was published would otherwise never grow one, and the video would
    // sit there saying how much of it is watchable with no way to watch it.
    // Rewritten only when it actually changes: replacing the anchor on every
    // poll would swallow a click that landed between ticks. The comparison is
    // against the last markup we wrote rather than against innerHTML, which
    // reads back normalised - `&middot;` returns as the character itself, so
    // comparing the two never matches and the anchor is replaced every tick.
    if (live) {
      const want = liveButton(s);
      if (live.dataset.live !== want) {
        live.dataset.live = want;
        live.innerHTML = want;
      }
    }
    el.dataset.status = s.status;
  }));

  if (finished) render();
}

/* ---------------------------------------------------------------- wire --- */
document.getElementById('btn-upload').onclick = openUpload;
document.getElementById('btn-link').onclick = openDownload;
window.__download = openDownload;
document.getElementById('btn-settings').onclick = openSettings;
window.__upload = openUpload;
window.__close = closeModal;

let searchTimer;
document.getElementById('search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    SEARCH = e.target.value.trim();
    if (!location.hash.startsWith('#/watch')) render();
  }, 220);
});

window.addEventListener('hashchange', render);

(async function boot() {
  const health = await api('/api/health').catch(() => null);
  if (health && !health.ffmpeg) {
    toast('ffmpeg is missing - run "python setup.py" in the project folder', true);
  }
  render();
  loadVoices();
})();
