import { STEM_COLORS, STEM_NAMES } from "./constants.js";
import { fmtTime } from "./utils.js";
import {
  footerTimeElapsed,
  footerTimeTotal,
  multitrack,
  npScrubFill,
  timeEl,
  totalDuration,
} from "./state.js";

const PREVIEW_POINTS = 960;
const STEM_SET = new Set(STEM_NAMES);

let labInitialized = false;
let labToken = 0;
let audioContext = null;
let previewRafId = null;
let renderPollId = null;
let playbackFocusMode = "waveform";

const previewPlayback = {
  nodes: [],
  offset: 0,
  startedAt: 0,
  playing: false,
  rafId: null,
};

const labState = {
  jobId: null,
  stems: [],
  buffers: new Map(),
  peaks: new Map(),
  rms: new Map(),
  loading: false,
  loaded: 0,
  error: "",
};

const controls = {};
const AUTO_FIELDS = ["space", "height", "focus", "vocal", "guard", "master"];
const UPMIX_TAB_KEY = "stemdeck.upmix.controller.tab";
const DEFAULT_STEM_SGI_CHECKPOINT = "weights/de_limiter_best.safetensors";

const AUTO_FIELD_CONFIG = {
  space: {
    param: "space",
    toInput: (value) => Math.round(clamp(value, 0, 1) * 100),
  },
  height: {
    param: "height",
    toInput: (value) => Math.round(clamp(value, 0, 1) * 100),
  },
  focus: {
    param: "focus",
    toInput: (value) => Math.round(clamp(value, 0, 1) * 100),
  },
  vocal: {
    param: "vocalGainDb",
    toInput: (value) => clamp(value, -6, 6).toFixed(1),
  },
  guard: {
    param: "guard",
    toInput: (value) => Math.round(clamp(value, 0, 1) * 100),
  },
  master: {
    param: "masterGain",
    toInput: (value) => Math.round(clamp(value, 0.25, 1.2) * 100),
  },
};

const ADVANCED_CLI_FIELDS = [
  { group: "tune", section: "Analysis", key: "analysis_backend", label: "Analysis", type: "select", default: "auto", options: ["auto", "numpy", "ffmpeg"] },
  { group: "tune", section: "Analysis", key: "silent_stem_filter", label: "Silent Filter", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Analysis", key: "window_ms", label: "Window ms", type: "number", default: 1000, step: 50 },
  { group: "tune", section: "Analysis", key: "hop_ms", label: "Hop ms", type: "number", default: 500, step: 50 },
  { group: "tune", section: "Focus", key: "focus_stem_mode", label: "Focus Mode", type: "select", default: "auto", options: ["auto", "manual", "off"] },
  { group: "tune", section: "Focus", key: "focus_stem", label: "Focus Stem", type: "select", default: "", options: ["", "vocals", "drums", "bass", "guitar", "piano", "other"] },
  { group: "tune", section: "Focus", key: "focus_vocal_min_share", label: "Vocal Min", type: "number", default: 0.05, step: 0.01 },
  { group: "tune", section: "Focus", key: "presence_priority", label: "Priority", type: "text", default: "", placeholder: "other,guitar,piano" },
  { group: "tune", section: "Focus", key: "priority_duck", label: "Priority Duck", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Focus", key: "priority_duck_max_db", label: "Duck dB", type: "number", default: 0.8, step: 0.05 },
  { group: "tune", section: "Focus", key: "temporal_duck_guard", label: "Duck Guard", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Focus", key: "temporal_duck_guard_strength", label: "Duck Guard Amt", type: "number", default: 0.75, step: 0.05 },
  { group: "tune", section: "Space", key: "space_bed_stems", label: "Space Stems", type: "text", default: "guitar,piano,other,drums" },
  { group: "tune", section: "Space", key: "stem_gain_mode", label: "Stem Gain", type: "select", default: "off", options: ["off", "spatial", "harman"] },
  { group: "tune", section: "Space", key: "stem_gain_limit_db", label: "Gain Limit", type: "number", default: 1.5, step: 0.1 },
  { group: "tune", section: "Space", key: "stem_aware_eq", label: "Stem EQ", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Space", key: "stem_aware_eq_limit_db", label: "EQ Limit", type: "number", default: 1.2, step: 0.1 },
  { group: "tune", section: "Space", key: "bass_fold_support", label: "Bass Fold", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Space", key: "bass_fold_support_target_db", label: "Bass Target", type: "number", default: -3.5, step: 0.1 },
  { group: "tune", section: "Space", key: "bass_fold_support_max_coeff", label: "Bass Coeff", type: "number", default: 0.12, step: 0.01 },
  { group: "tune", section: "Temporal", key: "temporal_remaster", label: "Remaster", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Temporal", key: "temporal_remaster_strength", label: "Remaster Amt", type: "number", default: 0.75, step: 0.05 },
  { group: "tune", section: "Temporal", key: "temporal_remaster_step_sec", label: "Step sec", type: "number", default: 4, step: 0.5 },
  { group: "tune", section: "Temporal", key: "temporal_mastering", label: "Mastering", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Temporal", key: "temporal_mastering_profile", label: "TM Profile", type: "select", default: "auto", options: ["auto", "conservative", "natural", "open"] },
  { group: "tune", section: "Temporal", key: "temporal_mastering_report", label: "TM Report", type: "path", default: "", placeholder: "/path/report.json" },
  { group: "tune", section: "Temporal", key: "temporal_feedback", label: "Feedback", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Temporal", key: "temporal_feedback_preflight", label: "FB Preflight", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "tune", section: "Temporal", key: "temporal_feedback_max_passes", label: "Feedback Passes", type: "number", default: 1, step: 1 },
  { group: "tune", section: "Temporal", key: "temporal_feedback_strength", label: "Feedback Amt", type: "number", default: 0.55, step: 0.05 },

  { group: "safety", section: "Recovery", key: "brickwall_recovery", label: "Brickwall", type: "select", default: "auto", options: ["auto", "off", "conservative", "natural", "open"] },
  { group: "safety", section: "Recovery", key: "brickwall_recovery_strength", label: "Brickwall Amt", type: "number", default: 0.75, step: 0.05 },
  { group: "safety", section: "Recovery", key: "brickwall_recovery_keep_stems", label: "Keep Recovery", type: "bool", default: false },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter", label: "Stem SGI", type: "select", default: "auto", options: ["off", "auto", "safe", "remaster", "repair", "sections2"] },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_checkpoint", label: "Stem CKPT", type: "path", default: DEFAULT_STEM_SGI_CHECKPOINT, placeholder: "weights/model.safetensors" },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_mix", label: "Stem Mix", type: "number", default: "", step: 0.05 },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_device", label: "Stem Device", type: "text", default: "auto" },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_chunk_sec", label: "Stem Chunk", type: "number", default: 8, step: 0.5 },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_overlap_sec", label: "Stem Overlap", type: "number", default: 1, step: 0.1 },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_peak_limit", label: "Stem Peak", type: "number", default: 0.98, step: 0.01 },
  { group: "safety", section: "Stem SGI", key: "stem_de_limiter_keep_stems", label: "Keep Stem SGI", type: "bool", default: false },
  { group: "safety", section: "Reference", key: "reference_guard", label: "Reference", type: "select", default: "auto", options: ["auto", "off", "original"] },
  { group: "safety", section: "Reference", key: "reference_match", label: "Ref Match", type: "select", default: "off", options: ["off", "original"] },
  { group: "safety", section: "Reference", key: "reference_match_limit_db", label: "Ref Limit", type: "number", default: 1, step: 0.1 },
  { group: "safety", section: "Reference", key: "reference_guard_tolerance_db", label: "Ref Tol", type: "number", default: 1, step: 0.1 },
  { group: "safety", section: "Reference", key: "style_reference", label: "Style Ref", type: "path", default: "", placeholder: "/path/song.wav" },
  { group: "safety", section: "Reference", key: "style_reference_strength", label: "Style Amt", type: "number", default: 0.35, step: 0.05 },
  { group: "safety", section: "Guards", key: "stem_presence_guard", label: "Presence", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Guards", key: "stem_presence_guard_limit_db", label: "Presence dB", type: "number", default: 0.6, step: 0.05 },
  { group: "safety", section: "Guards", key: "drum_dominance_guard", label: "Drum Guard", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Guards", key: "drum_dominance_guard_limit_db", label: "Drum dB", type: "number", default: 0.35, step: 0.05 },
  { group: "safety", section: "Guards", key: "bass_drum_guard", label: "Bass/Drum", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Guards", key: "bass_drum_guard_max_db", label: "Bass/Drum dB", type: "number", default: 1.2, step: 0.1 },
  { group: "safety", section: "Post Render", key: "post_render_analysis", label: "Post Analysis", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Post Render", key: "post_render_correct", label: "Post Correct", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Post Render", key: "post_render_correct_iterations", label: "Iterations", type: "number", default: 2, step: 1 },
  { group: "safety", section: "Post Render", key: "post_render_correct_tolerance_db", label: "Tolerance", type: "number", default: 0.55, step: 0.05 },
  { group: "safety", section: "Post Render", key: "post_render_correct_limit_db", label: "Limit dB", type: "number", default: 0.7, step: 0.05 },
  { group: "safety", section: "Post Render", key: "post_render_correct_space_limit", label: "Space Limit", type: "number", default: 0.06, step: 0.01 },
  { group: "safety", section: "Post Render", key: "post_render_correct_eq", label: "Post EQ", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "safety", section: "Post Render", key: "post_render_correct_eq_limit_db", label: "Post EQ dB", type: "number", default: 0.8, step: 0.05 },

  { group: "render", section: "Outputs", key: "flac_compression_level", label: "FLAC Level", type: "number", default: 8, step: 1 },
  { group: "render", section: "Outputs", key: "skip_flac", label: "Skip 5.1", type: "bool", default: false },
  { group: "render", section: "Outputs", key: "skip_stereo", label: "Skip Stereo", type: "bool", default: false },
  { group: "render", section: "Outputs", key: "render_apple_tv", label: "Apple TV MP4", type: "bool", default: false },
  { group: "render", section: "Outputs", key: "skip_bed", label: "Skip Bed WAV", type: "bool", default: false },
  { group: "render", section: "Outputs", key: "apple_tv_bitrate", label: "Apple Bitrate", type: "text", default: "768k" },
  { group: "render", section: "Stereo", key: "stereo_normalize", label: "Stereo Norm", type: "select", default: "linear", options: ["loudnorm", "linear", "off"] },
  { group: "render", section: "Stereo", key: "stereo_loudness_i", label: "Stereo LUFS", type: "number", default: -14, step: 0.5 },
  { group: "render", section: "Stereo", key: "stereo_loudness_tp", label: "Stereo TP", type: "number", default: -1, step: 0.1 },
  { group: "render", section: "Stereo", key: "stereo_loudness_lra", label: "Stereo LRA", type: "number", default: 11, step: 0.5 },
  { group: "render", section: "Stereo", key: "stereo_lfe_fold_gain", label: "LFE Fold", type: "number", default: 0, step: 0.05 },
  { group: "render", section: "Envelope", key: "stereo_envelope_match", label: "Envelope", type: "select", default: "auto", options: ["auto", "off"] },
  { group: "render", section: "Envelope", key: "stereo_envelope_window_sec", label: "Env Window", type: "number", default: 12, step: 1 },
  { group: "render", section: "Envelope", key: "stereo_envelope_hop_sec", label: "Env Hop", type: "number", default: 3, step: 0.5 },
  { group: "render", section: "Envelope", key: "stereo_envelope_smooth_sec", label: "Env Smooth", type: "number", default: 24, step: 1 },
  { group: "render", section: "Envelope", key: "stereo_envelope_strength", label: "Env Amt", type: "number", default: 0.75, step: 0.05 },
  { group: "render", section: "Envelope", key: "stereo_envelope_max_boost_db", label: "Env Boost", type: "number", default: 2.5, step: 0.1 },
  { group: "render", section: "Envelope", key: "stereo_envelope_max_cut_db", label: "Env Cut", type: "number", default: 1.5, step: 0.1 },
  { group: "render", section: "Stereo SGI", key: "de_limiter_checkpoint", label: "Stereo CKPT", type: "path", default: "", placeholder: "/path/model.ckpt" },
  { group: "render", section: "Stereo SGI", key: "de_limiter_mix", label: "Stereo Mix", type: "number", default: 1, step: 0.05 },
  { group: "render", section: "Stereo SGI", key: "de_limiter_device", label: "Stereo Device", type: "text", default: "auto" },
  { group: "render", section: "Stereo SGI", key: "de_limiter_chunk_sec", label: "Stereo Chunk", type: "number", default: 8, step: 0.5 },
  { group: "render", section: "Stereo SGI", key: "de_limiter_overlap_sec", label: "Stereo Overlap", type: "number", default: 1, step: 0.1 },
  { group: "render", section: "Stereo SGI", key: "de_limiter_peak_limit", label: "Stereo Peak", type: "number", default: 0.98, step: 0.01 },
  { group: "render", section: "Surround", key: "surround_normalize", label: "Surround Norm", type: "select", default: "loudnorm", options: ["loudnorm", "off"] },
  { group: "render", section: "Surround", key: "surround_loudness_i", label: "Surround LUFS", type: "number", default: -14, step: 0.5 },
  { group: "render", section: "Surround", key: "surround_loudness_tp", label: "Surround TP", type: "number", default: -2.5, step: 0.1 },
  { group: "render", section: "Surround", key: "surround_loudness_lra", label: "Surround LRA", type: "number", default: 11, step: 0.5 },
  { group: "render", section: "Surround", key: "surround_render_backend", label: "5.1 Render", type: "select", default: "native-wav", options: ["native-wav", "ffmpeg-flac"] },
  { group: "render", section: "Surround", key: "loudnorm_measure_backend", label: "Meter", type: "select", default: "native", options: ["native", "ffmpeg"] },
];

function formatBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n <= 0) return "";
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function linkLabel(output) {
  const label = output.label || output.kind || "Output";
  const size = formatBytes(output.size_bytes);
  return size ? `${label} · ${size}` : label;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function dbToAmp(db) {
  return Math.pow(10, db / 20);
}

function emitPlaybackFocus() {
  window.dispatchEvent(new CustomEvent("stemdeck:playback-focus-change", {
    detail: { mode: playbackFocusMode },
  }));
}

function emitPreviewState() {
  syncPlaybackFocusUi();
  window.dispatchEvent(new CustomEvent("stemdeck:upmix-preview-state", {
    detail: {
      playing: previewPlayback.playing,
      time: getUpmixPreviewTime(),
      duration: getUpmixPreviewDuration(),
    },
  }));
}

function syncPlaybackFocusUi() {
  const app = document.querySelector(".app");
  app?.setAttribute("data-playback-focus", playbackFocusMode);
  for (const mode of ["waveform", "upmix"]) {
    document
      .querySelector(`.widget[data-widget="${mode}"]`)
      ?.classList.toggle("playback-focus", playbackFocusMode === mode);
  }
  document
    .querySelector('.widget[data-widget="upmix"]')
    ?.classList.toggle("preview-active", previewPlayback.playing);
}

export function getPlaybackFocusMode() {
  return playbackFocusMode;
}

export function isUpmixPlaybackFocused() {
  return playbackFocusMode === "upmix";
}

export function isUpmixPreviewPlaying() {
  return previewPlayback.playing;
}

export function getUpmixPreviewDuration() {
  const bufferDurations = [...labState.buffers.values()].map((buffer) => buffer.duration || 0);
  const knownDuration = Math.max(0, totalDuration || 0, multitrack?.getDuration?.() || 0);
  return Math.max(knownDuration, ...bufferDurations, 0);
}

export function getUpmixPreviewTime() {
  const dur = getUpmixPreviewDuration();
  if (!previewPlayback.playing || !audioContext) {
    return clamp(previewPlayback.offset, 0, dur || Number.MAX_SAFE_INTEGER);
  }
  const elapsed = audioContext.currentTime - previewPlayback.startedAt;
  return clamp(previewPlayback.offset + elapsed, 0, dur || Number.MAX_SAFE_INTEGER);
}

function stopPreviewNodes() {
  const nodes = previewPlayback.nodes;
  previewPlayback.nodes = [];
  for (const { source, gain } of nodes) {
    try {
      source.onended = null;
      source.stop(0);
    } catch { /* already stopped */ }
    try { source.disconnect(); } catch { /* ignore */ }
    try { gain.disconnect(); } catch { /* ignore */ }
  }
}

function stopPreviewClock() {
  if (previewPlayback.rafId) cancelAnimationFrame(previewPlayback.rafId);
  previewPlayback.rafId = null;
}

function updatePreviewGains() {
  if (!previewPlayback.nodes.length) return;
  const p = upmixParams();
  for (const node of previewPlayback.nodes) {
    const next = stemFoldGain(node.stemName, p);
    try {
      node.gain.gain.setTargetAtTime(next, audioContext?.currentTime || 0, 0.015);
    } catch {
      node.gain.gain.value = next;
    }
  }
}

function updatePreviewTransportTime() {
  if (playbackFocusMode !== "upmix") return;
  const t = getUpmixPreviewTime();
  const dur = getUpmixPreviewDuration();
  if (timeEl) timeEl.textContent = `${fmtTime(t)} / ${fmtTime(dur)}`;
  if (footerTimeElapsed) footerTimeElapsed.textContent = fmtTime(t);
  if (footerTimeTotal) footerTimeTotal.textContent = fmtTime(dur);
  if (npScrubFill) {
    const pct = dur > 0 ? clamp((t / dur) * 100, 0, 100) : 0;
    npScrubFill.style.width = `${pct}%`;
  }
}

function startPreviewClock() {
  if (previewPlayback.rafId) return;
  const tick = () => {
    previewPlayback.rafId = null;
    updatePreviewTransportTime();
    drawPreview();
    const dur = getUpmixPreviewDuration();
    if (previewPlayback.playing && dur > 0 && getUpmixPreviewTime() >= dur - 0.02) {
      pauseUpmixPreview();
      previewPlayback.offset = dur;
      updatePreviewTransportTime();
      emitPreviewState();
      return;
    }
    if (previewPlayback.playing) previewPlayback.rafId = requestAnimationFrame(tick);
  };
  previewPlayback.rafId = requestAnimationFrame(tick);
}

async function ensureAudioContext() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) throw new Error("Web Audio is not available");
  audioContext ??= new AudioCtx();
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  return audioContext;
}

function setMultitrackTime(sec) {
  if (!multitrack) return;
  try {
    multitrack.setTime(clamp(sec, 0, multitrack.getDuration?.() || totalDuration || sec));
  } catch { /* ignore */ }
}

export function pauseUpmixPreview({ emit = true } = {}) {
  if (previewPlayback.playing) {
    previewPlayback.offset = getUpmixPreviewTime();
  }
  previewPlayback.playing = false;
  stopPreviewNodes();
  stopPreviewClock();
  updatePreviewTransportTime();
  drawPreview();
  if (emit) emitPreviewState();
}

export function stopUpmixPreview({ emit = true } = {}) {
  previewPlayback.playing = false;
  previewPlayback.offset = 0;
  stopPreviewNodes();
  stopPreviewClock();
  updatePreviewTransportTime();
  drawPreview();
  if (emit) emitPreviewState();
}

export async function playUpmixPreview(offset = getUpmixPreviewTime()) {
  initUpmixLab();
  if (!labState.buffers.size) {
    setLabStatus(labState.loading ? "Loading" : "Unavailable");
    return false;
  }

  const ctx = await ensureAudioContext();
  const dur = getUpmixPreviewDuration();
  const startOffset = dur > 0 && offset >= dur - 0.02 ? 0 : clamp(offset, 0, dur || offset);
  stopPreviewNodes();
  previewPlayback.offset = startOffset;
  previewPlayback.startedAt = ctx.currentTime;
  previewPlayback.playing = true;

  const p = upmixParams();
  for (const stem of labState.stems) {
    const buffer = labState.buffers.get(stem.name);
    if (!buffer) continue;
    if (startOffset >= Math.max(0, buffer.duration - 0.01)) continue;
    const source = ctx.createBufferSource();
    const gain = ctx.createGain();
    source.buffer = buffer;
    gain.gain.value = stemFoldGain(stem.name, p);
    source.connect(gain).connect(ctx.destination);
    try {
      source.start(0, Math.max(0, startOffset));
      previewPlayback.nodes.push({ stemName: stem.name, source, gain });
    } catch {
      try { source.disconnect(); } catch { /* ignore */ }
      try { gain.disconnect(); } catch { /* ignore */ }
    }
  }

  if (!previewPlayback.nodes.length) {
    stopUpmixPreview();
    return false;
  }
  setLabStatus("Preview");
  updatePreviewTransportTime();
  startPreviewClock();
  emitPreviewState();
  return true;
}

export function setUpmixPreviewTime(sec) {
  const dur = getUpmixPreviewDuration();
  const next = clamp(sec, 0, dur || sec);
  previewPlayback.offset = next;
  if (previewPlayback.playing) {
    playUpmixPreview(next).catch(() => {
      pauseUpmixPreview();
      setLabStatus("Unavailable");
    });
  } else {
    updatePreviewTransportTime();
    drawPreview();
    emitPreviewState();
  }
}

export function seekUpmixPreview(deltaSec) {
  setUpmixPreviewTime(getUpmixPreviewTime() + deltaSec);
}

export function setPlaybackFocusMode(mode, { preservePlayback = true } = {}) {
  const next = mode === "upmix" ? "upmix" : "waveform";
  if (next === playbackFocusMode) {
    syncPlaybackFocusUi();
    return;
  }

  const previous = playbackFocusMode;
  const wasPreviewPlaying = previewPlayback.playing;
  const wasMultitrackPlaying = Boolean(multitrack?.isPlaying?.());
  const handoffTime = previous === "upmix"
    ? getUpmixPreviewTime()
    : (multitrack?.getCurrentTime?.() || previewPlayback.offset || 0);

  playbackFocusMode = next;
  syncPlaybackFocusUi();
  emitPlaybackFocus();

  if (next === "upmix") {
    previewPlayback.offset = handoffTime;
    if (wasMultitrackPlaying) {
      try { multitrack.pause(); } catch { /* ignore */ }
      if (preservePlayback) {
        playUpmixPreview(handoffTime).catch(() => {
          pauseUpmixPreview();
          setLabStatus("Unavailable");
        });
      }
    } else {
      updatePreviewTransportTime();
      drawPreview();
    }
    return;
  }

  if (previous === "upmix") {
    pauseUpmixPreview({ emit: false });
    setMultitrackTime(handoffTime);
    if (preservePlayback && wasPreviewPlaying && multitrack) {
      const ctx = multitrack.audioContext;
      if (ctx && ctx.state === "suspended") ctx.resume().catch(() => {});
      try { multitrack.play(); } catch { /* ignore */ }
    }
  }
  emitPreviewState();
}

function wirePlaybackFocus() {
  const bind = (selector, mode) => {
    const el = document.querySelector(selector);
    if (!el) return;
    el.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      setPlaybackFocusMode(mode);
    }, true);
    el.addEventListener("focusin", () => setPlaybackFocusMode(mode), true);
  };
  bind('.widget[data-widget="waveform"]', "waveform");
  bind('.widget[data-widget="upmix"]', "upmix");
  syncPlaybackFocusUi();
}

function setUpmixControllerTab(tab, { focusTab = false, persist = true } = {}) {
  const panels = [...document.querySelectorAll("[data-upmix-panel]")];
  const tabs = [...document.querySelectorAll("[data-upmix-tab]")];
  const valid = tabs.some((item) => item.dataset.upmixTab === tab);
  const next = valid ? tab : "tune";
  for (const btn of tabs) {
    const active = btn.dataset.upmixTab === next;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
    btn.tabIndex = active ? 0 : -1;
    if (active && focusTab) btn.focus();
  }
  for (const panel of panels) {
    const active = panel.dataset.upmixPanel === next;
    panel.classList.toggle("active", active);
    panel.toggleAttribute("hidden", !active);
  }
  if (persist) {
    try { localStorage.setItem(UPMIX_TAB_KEY, next); } catch { /* ignore */ }
  }
}

function wireUpmixControllerTabs() {
  const tabs = [...document.querySelectorAll("[data-upmix-tab]")];
  if (!tabs.length) return;
  for (const btn of tabs) {
    btn.addEventListener("click", () => setUpmixControllerTab(btn.dataset.upmixTab));
    btn.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.code)) return;
      event.preventDefault();
      const current = tabs.indexOf(btn);
      let nextIndex = current;
      if (event.code === "Home") nextIndex = 0;
      else if (event.code === "End") nextIndex = tabs.length - 1;
      else nextIndex = (current + (event.code === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      setUpmixControllerTab(tabs[nextIndex].dataset.upmixTab, { focusTab: true });
    });
  }
  let initial = "tune";
  try {
    const saved = localStorage.getItem(UPMIX_TAB_KEY);
    if (tabs.some((item) => item.dataset.upmixTab === saved)) initial = saved;
  } catch { /* ignore */ }
  setUpmixControllerTab(initial, { persist: false });
}

function makeAdvancedInput(field) {
  if (field.type === "select") {
    const select = document.createElement("select");
    select.dataset.upmixAdvanced = field.key;
    for (const optionValue of field.options || []) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue || "auto";
      select.appendChild(option);
    }
    select.value = String(field.default ?? "");
    return select;
  }
  const input = document.createElement("input");
  input.dataset.upmixAdvanced = field.key;
  if (field.type === "bool") {
    input.type = "checkbox";
    input.checked = Boolean(field.default);
    input.className = "upmix-auto-toggle";
    return input;
  }
  input.type = field.type === "number" ? "number" : "text";
  if (field.step !== undefined) input.step = String(field.step);
  if (field.placeholder) input.placeholder = field.placeholder;
  if (field.default !== undefined && field.default !== null) input.value = String(field.default);
  return input;
}

function renderAdvancedControls() {
  for (const root of document.querySelectorAll("[data-upmix-extra-group]")) {
    const group = root.dataset.upmixExtraGroup;
    root.textContent = "";
    let currentSection = "";
    for (const field of ADVANCED_CLI_FIELDS.filter((item) => item.group === group)) {
      if (field.section !== currentSection) {
        currentSection = field.section;
        const heading = document.createElement("h4");
        heading.className = "upmix-section-title";
        heading.textContent = currentSection;
        root.appendChild(heading);
      }
      const label = document.createElement("label");
      label.className = field.type === "bool"
        ? "upmix-advanced-control upmix-boolean-control"
        : "upmix-advanced-control";
      const span = document.createElement("span");
      span.textContent = field.label;
      label.appendChild(span);
      label.appendChild(makeAdvancedInput(field));
      root.appendChild(label);
    }
  }
}

function collectAdvancedOptions() {
  const advanced = {};
  const stemSgiMode = String(
    document.querySelector('[data-upmix-advanced="stem_de_limiter"]')?.value ?? "off",
  ).trim();
  for (const field of ADVANCED_CLI_FIELDS) {
    const el = document.querySelector(`[data-upmix-advanced="${field.key}"]`);
    if (!el) continue;
    if (field.type === "bool") {
      if (field.key === "render_apple_tv") {
        advanced.skip_apple_tv = !el.checked;
        continue;
      }
      if (el.checked) advanced[field.key] = true;
      continue;
    }
    const raw = String(el.value ?? "").trim();
    if (field.key === "stem_de_limiter_checkpoint") {
      if (stemSgiMode === "off" || stemSgiMode === "safe") continue;
      advanced[field.key] = raw || DEFAULT_STEM_SGI_CHECKPOINT;
      continue;
    }
    if (!raw) continue;
    advanced[field.key] = field.type === "number" ? Number(raw) : raw;
  }
  return advanced;
}

function rangeNumber(el, fallback) {
  const value = Number(el?.value);
  return Number.isFinite(value) ? value : fallback;
}

function autoEnabled(field) {
  return Boolean(controls[`${field}Auto`]?.checked);
}

function stemStats() {
  const energy = {};
  let totalEnergy = 0;
  let peakAbs = 0;
  for (const stemName of STEM_NAMES) {
    const rms = labState.rms.get(stemName) || 0;
    const e = rms * rms;
    energy[stemName] = e;
    totalEnergy += e;
    const peaks = labState.peaks.get(stemName);
    if (peaks) {
      for (const [mn, mx] of peaks) {
        peakAbs = Math.max(peakAbs, Math.abs(mn), Math.abs(mx));
      }
    }
  }
  const share = (stemName, fallback = 0) => (
    totalEnergy > 1e-10 ? (energy[stemName] || 0) / totalEnergy : fallback
  );
  return {
    totalEnergy,
    peakAbs,
    vocalShare: share("vocals", 0.18),
    drumsShare: share("drums", 0.22),
    bassShare: share("bass", 0.14),
    guitarShare: share("guitar", 0.12),
    pianoShare: share("piano", 0.10),
    otherShare: share("other", 0.24),
  };
}

function estimateFoldMax(p) {
  let maxAbs = 0;
  const params = { ...p, masterGain: 1 };
  for (const stem of labState.stems) {
    const peaks = labState.peaks.get(stem.name);
    if (!peaks) continue;
    const gain = stemFoldGain(stem.name, params);
    for (const [mn, mx] of peaks) {
      maxAbs = Math.max(maxAbs, Math.abs(mn * gain), Math.abs(mx * gain));
    }
  }
  return maxAbs;
}

function readManualParams() {
  return {
    profile: controls.profile?.value || "auto",
    lfeModeRaw: controls.lfeMode?.value || "auto",
    lfeMode: controls.lfeMode?.value || "auto",
    space: Number(controls.space?.value || 0) / 100,
    height: Number(controls.height?.value || 0) / 100,
    focus: Number(controls.focus?.value || 0) / 100,
    vocalGainDb: rangeNumber(controls.vocal, 0),
    guard: Number(controls.guard?.value || 0) / 100,
    masterGain: Number(controls.master?.value || 84) / 100,
    autoModes: [],
  };
}

function computeAutoValues(base) {
  const stats = stemStats();
  const wideShare = stats.guitarShare + stats.pianoShare + stats.otherShare;
  const supportShare = wideShare + stats.drumsShare * 0.45;
  const space = clamp(0.30 + supportShare * 0.68 - stats.vocalShare * 0.12, 0.24, 0.88);
  const height = clamp(0.18 + wideShare * 0.46 + space * 0.22 - stats.bassShare * 0.10, 0.16, 0.78);
  const focus = clamp(0.46 + stats.vocalShare * 0.62 - wideShare * 0.10, 0.38, 0.90);
  const vocalGainDb = clamp((0.16 - stats.vocalShare) * 8.0, -1.8, 2.4);
  const guard = clamp(0.54 + clamp(stats.peakAbs - 0.65, 0, 0.45) * 0.64 + stats.vocalShare * 0.08, 0.48, 0.92);
  const lfeEnergy = stats.bassShare + stats.drumsShare * 0.42;
  const lfeMode = lfeEnergy > 0.34 ? "normal" : lfeEnergy > 0.08 ? "light" : "off";
  const withoutMaster = {
    ...base,
    space: autoEnabled("space") ? space : base.space,
    height: autoEnabled("height") ? height : base.height,
    focus: autoEnabled("focus") ? focus : base.focus,
    vocalGainDb: autoEnabled("vocal") ? vocalGainDb : base.vocalGainDb,
    guard: autoEnabled("guard") ? guard : base.guard,
    lfeMode: base.lfeModeRaw === "auto" ? lfeMode : base.lfeModeRaw,
    masterGain: 1,
  };
  const foldMax = estimateFoldMax(withoutMaster);
  const masterGain = foldMax > 0 ? clamp(0.86 / foldMax, 0.62, 1.04) : 0.84;
  return { space, height, focus, vocalGainDb, guard, masterGain, lfeMode };
}

function applyAutoValuesToControls(values) {
  for (const field of AUTO_FIELDS) {
    if (!autoEnabled(field)) continue;
    const cfg = AUTO_FIELD_CONFIG[field];
    const el = controls[field];
    if (!cfg || !el) continue;
    el.value = String(cfg.toInput(values[cfg.param]));
  }
}

function upmixParams() {
  initUpmixLab();
  const base = readManualParams();
  const auto = computeAutoValues(base);
  applyAutoValuesToControls(auto);
  const p = readManualParams();
  const autoModes = [];
  for (const field of AUTO_FIELDS) {
    if (autoEnabled(field)) autoModes.push(field);
  }
  if (p.lfeModeRaw === "auto") autoModes.push("lfe");
  p.lfeMode = p.lfeModeRaw === "auto" ? auto.lfeMode : p.lfeModeRaw;
  p.autoModes = autoModes;
  return p;
}

function renderPayload() {
  const p = upmixParams();
  return {
    profile: p.profile,
    lfe_mode: p.lfeMode,
    vocal_gain_db: p.vocalGainDb,
    space_bed_strength: p.space,
    temporal_mastering_strength: clamp(0.35 + p.height * 0.55, 0, 1),
    stem_aware_eq_strength: p.guard,
    master_gain: p.masterGain,
    auto_modes: p.autoModes,
    advanced: collectAdvancedOptions(),
  };
}

function setLabStatus(text) {
  const el = document.getElementById("upmixLabStatus");
  if (el) el.textContent = text;
}

function updateControlOutputs() {
  const p = upmixParams();
  const set = (name, value) => {
    const el = document.querySelector(`[data-upmix-output="${name}"]`);
    if (el) el.textContent = value;
  };
  set("space", String(Math.round(p.space * 100)));
  set("height", String(Math.round(p.height * 100)));
  set("focus", String(Math.round(p.focus * 100)));
  set("vocal", `${p.vocalGainDb.toFixed(1)} dB`);
  set("guard", String(Math.round(p.guard * 100)));
  set("master", p.masterGain.toFixed(2));
}

function lfeSendAmount(stemName, mode) {
  if (mode === "auto") return lfeSendAmount(stemName, computeAutoValues(readManualParams()).lfeMode);
  if (mode === "off") return 0;
  const light = mode === "light";
  if (stemName === "bass") return light ? 0.08 : 0.16;
  if (stemName === "drums") return light ? 0.035 : 0.07;
  return 0;
}

function stemSends(stemName, p) {
  const forward = p.profile === "front51";
  const space = forward ? p.space * 0.58 : p.space;
  const height = p.height;
  const focus = p.focus;
  const guard = p.guard;
  const lfe = lfeSendAmount(stemName, p.lfeMode);
  const bedGuard = 1 - guard * focus * 0.18;

  if (stemName === "vocals") {
    return {
      front: 0.48 + 0.06 * (1 - focus),
      center: 0.16 + 0.38 * focus,
      side: 0.025 + 0.055 * space,
      rear: 0.008 * space,
      height: 0.012 + 0.055 * height,
      lfe: 0,
    };
  }
  if (stemName === "drums") {
    return {
      front: 0.66,
      center: 0.025 + 0.035 * (1 - focus),
      side: (0.12 + 0.22 * space) * bedGuard,
      rear: (0.035 + 0.08 * space) * bedGuard,
      height: (0.04 + 0.13 * height) * bedGuard,
      lfe,
    };
  }
  if (stemName === "bass") {
    return {
      front: 0.20,
      center: 0.18 + 0.18 * focus,
      side: 0.018 * space,
      rear: 0,
      height: 0,
      lfe,
    };
  }
  if (stemName === "other") {
    return {
      front: Math.max(0.12, 0.26 - 0.07 * space),
      center: 0.006,
      side: (0.15 + 0.26 * space) * bedGuard,
      rear: (0.06 + 0.25 * space) * bedGuard,
      height: (0.025 + 0.14 * height) * bedGuard,
      lfe: 0,
    };
  }
  const piano = stemName === "piano";
  return {
    front: piano ? 0.45 : 0.50,
    center: piano ? 0.015 : 0.006,
    side: (0.08 + 0.20 * space) * bedGuard,
    rear: (0.025 + 0.12 * space) * bedGuard,
    height: (0.02 + 0.12 * height) * bedGuard,
    lfe: 0,
  };
}

function stemFoldGain(stemName, p) {
  const sends = stemSends(stemName, p);
  const fold =
    sends.front
    + sends.center * 0.707
    + (sends.side + sends.rear) * 0.62
    + sends.height * 0.46
    + sends.lfe * 0.18;
  const vocal = stemName === "vocals" ? dbToAmp(p.vocalGainDb) : 1;
  return fold * vocal * p.masterGain;
}

function readMonoSample(buffer, index) {
  if (buffer.numberOfChannels <= 1) return buffer.getChannelData(0)[index] || 0;
  let sum = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    sum += buffer.getChannelData(ch)[index] || 0;
  }
  return sum / buffer.numberOfChannels;
}

function bufferPreviewPeaks(buffer) {
  const count = PREVIEW_POINTS;
  const binSize = Math.max(1, Math.floor(buffer.length / count));
  const peaks = new Array(count);
  let sumSq = 0;
  let n = 0;
  for (let i = 0; i < count; i++) {
    const start = i * binSize;
    const end = i === count - 1 ? buffer.length : Math.min(buffer.length, start + binSize);
    let mn = 0;
    let mx = 0;
    for (let j = start; j < end; j++) {
      const v = readMonoSample(buffer, j);
      if (v > mx) mx = v;
      else if (v < mn) mn = v;
      sumSq += v * v;
      n += 1;
    }
    peaks[i] = [mn, mx];
  }
  return { peaks, rms: Math.sqrt(sumSq / Math.max(1, n)) };
}

function combinedPeaks(p) {
  const points = PREVIEW_POINTS;
  const out = Array.from({ length: points }, () => [0, 0]);
  let maxAbs = 0;
  for (const stem of labState.stems) {
    const peaks = labState.peaks.get(stem.name);
    if (!peaks) continue;
    const gain = stemFoldGain(stem.name, p);
    for (let i = 0; i < points; i++) {
      out[i][0] += peaks[i][0] * gain;
      out[i][1] += peaks[i][1] * gain;
    }
  }
  for (const [mn, mx] of out) maxAbs = Math.max(maxAbs, Math.abs(mn), Math.abs(mx));
  return { peaks: out, maxAbs };
}

function drawWaveShape(ctx, peaks, width, height, norm, color, alpha = 1) {
  const mid = height / 2;
  ctx.beginPath();
  for (let i = 0; i < peaks.length; i++) {
    const x = (i / (peaks.length - 1)) * width;
    const y = mid - clamp(peaks[i][1] * norm, -1, 1) * (height * 0.43);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = peaks.length - 1; i >= 0; i--) {
    const x = (i / (peaks.length - 1)) * width;
    const y = mid - clamp(peaks[i][0] * norm, -1, 1) * (height * 0.43);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function drawPreview() {
  const canvas = document.getElementById("upmixPreviewCanvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width || 640);
  const height = Math.max(120, rect.height || 168);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "rgba(17, 25, 32, 0.96)");
  bg.addColorStop(1, "rgba(10, 16, 22, 0.96)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.11)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 6; i++) {
    const x = (width / 6) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(226, 232, 240, 0.18)";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (labState.error) {
    ctx.fillStyle = "rgba(240, 180, 170, 0.9)";
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.fillText(labState.error, 18, height / 2);
    return;
  }
  if (labState.loading || labState.peaks.size === 0) {
    ctx.fillStyle = "rgba(199, 221, 245, 0.78)";
    ctx.font = "600 13px system-ui, sans-serif";
    const count = labState.stems.length || STEM_NAMES.length;
    ctx.fillText(`Loading ${labState.loaded}/${count}`, 18, height / 2);
    return;
  }

  const p = upmixParams();
  const { peaks, maxAbs } = combinedPeaks(p);
  const norm = maxAbs > 0 ? Math.min(3.2, 0.92 / maxAbs) : 1;
  for (const stem of labState.stems) {
    const stemPeaks = labState.peaks.get(stem.name);
    if (!stemPeaks) continue;
    const gain = stemFoldGain(stem.name, p);
    const weighted = stemPeaks.map(([mn, mx]) => [mn * gain, mx * gain]);
    drawWaveShape(ctx, weighted, width, height, norm, STEM_COLORS[stem.name] || "#8aa0ad", 0.10);
  }
  drawWaveShape(ctx, peaks, width, height, norm, "rgba(126, 196, 255, 0.78)", 1);

  if (maxAbs > 1) {
    ctx.fillStyle = "rgba(232, 95, 111, 0.14)";
    ctx.fillRect(0, 0, width, 8);
    ctx.fillRect(0, height - 8, width, 8);
  }

  const t = playbackFocusMode === "upmix"
    ? getUpmixPreviewTime()
    : (multitrack?.getCurrentTime?.() || 0);
  const dur = playbackFocusMode === "upmix"
    ? getUpmixPreviewDuration()
    : (totalDuration || multitrack?.getDuration?.() || 0);
  if (dur > 0) {
    const x = clamp(t / dur, 0, 1) * width;
    ctx.strokeStyle = "rgba(216, 168, 74, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function updateMeters() {
  const p = upmixParams();
  const totals = { front: 0, center: 0, side: 0, rear: 0, height: 0, lfe: 0 };
  for (const stem of labState.stems) {
    const rms = labState.rms.get(stem.name) || 0;
    const vocal = stem.name === "vocals" ? dbToAmp(p.vocalGainDb) : 1;
    const sends = stemSends(stem.name, p);
    for (const key of Object.keys(totals)) {
      totals[key] += rms * sends[key] * vocal * p.masterGain;
    }
  }
  const max = Math.max(0.0001, ...Object.values(totals));
  for (const [key, value] of Object.entries(totals)) {
    const el = document.querySelector(`[data-upmix-meter="${key}"]`);
    el?.style.setProperty("--meter", `${Math.round((value / max) * 100)}%`);
  }
}

function refreshPreview() {
  updateControlOutputs();
  updateMeters();
  updatePreviewGains();
  drawPreview();
}

function startPreviewRaf() {
  if (previewRafId) return;
  const tick = () => {
    previewRafId = null;
    if (!document.getElementById("upmixLab")?.classList.contains("hidden")) {
      drawPreview();
      previewRafId = requestAnimationFrame(tick);
    }
  };
  previewRafId = requestAnimationFrame(tick);
}

function stopPreviewRaf() {
  if (previewRafId) cancelAnimationFrame(previewRafId);
  previewRafId = null;
}

function stopRenderPoll() {
  if (renderPollId) clearInterval(renderPollId);
  renderPollId = null;
}

async function pollRenderState(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!res.ok) return;
    const state = await res.json();
    renderUpmixPanel(state);
    if (state.status !== "upmixing") {
      stopRenderPoll();
      setLabStatus(state.upmix_error ? "Failed" : "Ready");
    }
  } catch {
    stopRenderPoll();
  }
}

function startRenderPoll(jobId) {
  stopRenderPoll();
  renderPollId = setInterval(() => pollRenderState(jobId), 1200);
  pollRenderState(jobId);
}

async function renderTunedUpmix() {
  if (!labState.jobId) return;
  const btn = document.getElementById("upmixRenderBtn");
  btn?.setAttribute("disabled", "true");
  setLabStatus("Rendering");
  try {
    const res = await fetch(`/api/jobs/${labState.jobId}/upmix/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(renderPayload()),
    });
    const state = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(state.detail || `Render failed: ${res.status}`);
    renderUpmixPanel(state);
    startRenderPoll(labState.jobId);
  } catch (err) {
    setLabStatus("Failed");
    labState.error = err?.message || "Render failed";
    drawPreview();
  } finally {
    btn?.removeAttribute("disabled");
  }
}

export function initUpmixLab() {
  if (labInitialized) return;
  renderAdvancedControls();
  controls.profile = document.getElementById("upmixProfile");
  controls.lfeMode = document.getElementById("upmixLfeMode");
  controls.space = document.getElementById("upmixSpace");
  controls.height = document.getElementById("upmixHeight");
  controls.focus = document.getElementById("upmixFocus");
  controls.vocal = document.getElementById("upmixVocal");
  controls.guard = document.getElementById("upmixGuard");
  controls.master = document.getElementById("upmixMaster");
  controls.spaceAuto = document.getElementById("upmixSpaceAuto");
  controls.heightAuto = document.getElementById("upmixHeightAuto");
  controls.focusAuto = document.getElementById("upmixFocusAuto");
  controls.vocalAuto = document.getElementById("upmixVocalAuto");
  controls.guardAuto = document.getElementById("upmixGuardAuto");
  controls.masterAuto = document.getElementById("upmixMasterAuto");
  for (const field of AUTO_FIELDS) {
    controls[field]?.addEventListener("input", (event) => {
      if (event.isTrusted && controls[`${field}Auto`]) {
        controls[`${field}Auto`].checked = false;
      }
    });
  }
  for (const el of Object.values(controls)) {
    el?.addEventListener("input", refreshPreview);
    el?.addEventListener("change", refreshPreview);
  }
  for (const el of document.querySelectorAll("[data-upmix-advanced]")) {
    el.addEventListener("input", refreshPreview);
    el.addEventListener("change", refreshPreview);
  }
  document.getElementById("upmixRenderBtn")?.addEventListener("click", renderTunedUpmix);
  window.addEventListener("resize", refreshPreview);
  wirePlaybackFocus();
  wireUpmixControllerTabs();
  labInitialized = true;
  updateControlOutputs();
}

async function decodeStem(stem, token) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) throw new Error("Web Audio is not available");
  audioContext ??= new AudioCtx();
  const res = await fetch(stem.url, { cache: "force-cache" });
  if (!res.ok) throw new Error(`${stem.name}: ${res.status}`);
  const data = await res.arrayBuffer();
  const buffer = await audioContext.decodeAudioData(data);
  if (token !== labToken) return;
  const { peaks, rms } = bufferPreviewPeaks(buffer);
  labState.buffers.set(stem.name, buffer);
  labState.peaks.set(stem.name, peaks);
  labState.rms.set(stem.name, rms);
  labState.loaded += 1;
  drawPreview();
}

export function prepareUpmixLab(jobId, stems = []) {
  initUpmixLab();
  labToken += 1;
  const token = labToken;
  stopRenderPoll();
  labState.jobId = jobId;
  labState.stems = stems.filter((stem) => STEM_SET.has(stem.name) && stem.url);
  labState.buffers = new Map();
  labState.peaks = new Map();
  labState.rms = new Map();
  labState.loading = labState.stems.length > 0;
  labState.loaded = 0;
  labState.error = "";
  stopUpmixPreview({ emit: false });

  const lab = document.getElementById("upmixLab");
  lab?.classList.toggle("hidden", labState.stems.length === 0);
  if (!labState.stems.length) {
    stopPreviewRaf();
    setLabStatus("Idle");
    return;
  }
  setLabStatus("Loading");
  drawPreview();
  Promise.allSettled(labState.stems.map((stem) => decodeStem(stem, token))).then((results) => {
    if (token !== labToken) return;
    labState.loading = false;
    const rejected = results.find((item) => item.status === "rejected");
    if (labState.peaks.size === 0 && rejected?.status === "rejected") {
      labState.error = rejected.reason?.message || "Preview unavailable";
      setLabStatus("Unavailable");
    } else {
      setLabStatus("Ready");
    }
    refreshPreview();
    startPreviewRaf();
  });
}

export function resetUpmixLab() {
  labToken += 1;
  labState.jobId = null;
  labState.stems = [];
  labState.buffers = new Map();
  labState.peaks = new Map();
  labState.rms = new Map();
  labState.loading = false;
  labState.loaded = 0;
  labState.error = "";
  stopUpmixPreview({ emit: false });
  setPlaybackFocusMode("waveform", { preservePlayback: false });
  stopRenderPoll();
  stopPreviewRaf();
  document.getElementById("upmixLab")?.classList.add("hidden");
  setLabStatus("Idle");
}

export function renderUpmixPanel(state = {}) {
  const card = document.getElementById("upmix-card");
  const status = document.getElementById("upmix-status");
  const links = document.getElementById("upmix-links");
  if (!card || !status || !links) return;

  const requested = Boolean(state.upmix_requested);
  const outputs = Array.isArray(state.upmix_outputs) ? state.upmix_outputs : [];
  const error = state.upmix_error || "";

  card.classList.toggle("hidden", !requested && outputs.length === 0 && !error);
  links.textContent = "";

  if (error) {
    status.textContent = "Failed";
    card.classList.add("has-error");
    const detail = document.createElement("small");
    detail.className = "upmix-error";
    detail.textContent = error;
    links.appendChild(detail);
    return;
  }

  card.classList.remove("has-error");
  if (outputs.length === 0) {
    status.textContent = requested ? "Queued" : "Off";
    return;
  }

  status.textContent = `${outputs.length} files`;
  for (const output of outputs) {
    if (!output?.url) continue;
    const link = document.createElement("a");
    link.className = "upmix-dl";
    link.href = output.url;
    link.download = output.filename || "";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = linkLabel(output);
    links.appendChild(link);
  }
}

export function resetUpmixPanel() {
  renderUpmixPanel({
    upmix_requested: false,
    upmix_outputs: [],
    upmix_error: null,
  });
}
