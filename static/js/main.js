import {
  playBtn, loopBtn, multitrack, loopEnabled, loopStart, loopEnd,
  setLoopStart, setLoopEnd, selectedStems, saveSelectedStems,
  upmixRequested, setUpmixRequested,
} from "./state.js";
import { STEM_NAMES } from "./constants.js";
import { renderEmptyShell, buildStripStems } from "./player.js";
import { wireJobForm } from "./job.js";
import {
  getActiveTransportTime,
  nudgeTransport,
  togglePlayPause,
  updateLoopRegionVisual,
  wireTransportButtons,
} from "./transport.js";
import { wireStemListControls, wireMixerToolbar } from "./mixer.js";
import { initCatalog } from "./catalog.js";
import { setPlaybackFocusMode } from "./upmix.js";

// ─── Stem choice toggles on the import page ───
//
// Filter-chip semantics (Spotify-style). The natural mental model when
// a user sees all 6 stems lit up is "everything is extracted"; when
// they then click ONE chip, they expect "now only this one". A plain
// toggle inverts the clicked chip and leaves the others on, which
// reads as "I just deselected the one I wanted" -- exactly the user
// confusion that prompted this fix.
//
// Algorithm:
//  - "All selected" is the implicit default (no filter applied).
//  - First click on a chip while in default state switches to
//    "only this stem" (clears all others).
//  - Subsequent clicks on inactive chips ADD them to the filter.
//  - Clicks on the only-selected chip clear it; if that empties the
//    selection, we revert to "all selected" (wraparound).
//
// Persisted across reloads so the next song honors the user's last
// chosen subset, but a 0-selection state is normalized to all 6.
function refreshStemChoiceVisuals() {
  for (const btn of document.querySelectorAll(".stem-choice[data-stem]")) {
    btn.setAttribute(
      "aria-pressed",
      String(selectedStems.has(btn.dataset.stem)),
    );
  }
}

function handleStemChoiceClick(stem) {
  const allSelected = selectedStems.size === STEM_NAMES.length;
  if (allSelected) {
    // Default state -> switch to "only this stem".
    selectedStems.clear();
    selectedStems.add(stem);
  } else if (selectedStems.has(stem)) {
    selectedStems.delete(stem);
    if (selectedStems.size === 0) {
      // Empty out wraps back to "all" so the user is never stuck.
      for (const n of STEM_NAMES) selectedStems.add(n);
    }
  } else {
    selectedStems.add(stem);
  }
  saveSelectedStems();
  refreshStemChoiceVisuals();
  buildStripStems();
}

function wireStemChoiceButtons() {
  refreshStemChoiceVisuals();
  for (const btn of document.querySelectorAll(".stem-choice[data-stem]")) {
    btn.addEventListener("click", () => handleStemChoiceClick(btn.dataset.stem));
  }
}

function refreshUpmixChoiceVisual() {
  const btn = document.getElementById("upmixChoice");
  if (!btn) return;
  btn.setAttribute("aria-pressed", String(upmixRequested));
}

function wireUpmixChoiceButton() {
  const btn = document.getElementById("upmixChoice");
  if (!btn) return;
  refreshUpmixChoiceVisual();
  btn.addEventListener("click", () => {
    setUpmixRequested(!upmixRequested);
    refreshUpmixChoiceVisual();
    buildStripStems();
  });
}

const WORKSPACE_TAB_KEY = "stemdeck.workspace.tab";

function setWorkspaceTab(tab, { focusTab = false, persist = true } = {}) {
  const next = tab === "upmix" ? "upmix" : "mix";
  const app = document.querySelector(".app");
  app?.setAttribute("data-workspace-tab", next);

  for (const btn of document.querySelectorAll("[data-workspace-tab]")) {
    const active = btn.dataset.workspaceTab === next;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
    btn.tabIndex = active ? 0 : -1;
    if (active && focusTab) btn.focus();
  }

  if (persist) {
    try { localStorage.setItem(WORKSPACE_TAB_KEY, next); } catch { /* ignore */ }
  }
  setPlaybackFocusMode(next === "upmix" ? "upmix" : "waveform");
}

function wireWorkspaceTabs() {
  const tabs = [...document.querySelectorAll("[data-workspace-tab]")];
  if (!tabs.length) return;
  for (const btn of tabs) {
    btn.addEventListener("click", () => setWorkspaceTab(btn.dataset.workspaceTab));
    btn.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.code)) return;
      event.preventDefault();
      const current = tabs.indexOf(btn);
      let nextIndex = current;
      if (event.code === "Home") nextIndex = 0;
      else if (event.code === "End") nextIndex = tabs.length - 1;
      else nextIndex = (current + (event.code === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      setWorkspaceTab(tabs[nextIndex].dataset.workspaceTab, { focusTab: true });
    });
  }
  let initial = "mix";
  try {
    const saved = localStorage.getItem(WORKSPACE_TAB_KEY);
    if (saved === "upmix" || saved === "mix") initial = saved;
  } catch { /* ignore */ }
  setWorkspaceTab(initial, { persist: false });
}

// ─── Wire everything up ───

wireJobForm();
wireTransportButtons();
wireStemListControls();
wireMixerToolbar();
wireStemChoiceButtons();
wireUpmixChoiceButton();
wireWorkspaceTabs();
initCatalog();
wireFileDrop();
wireAppShellControls();

// ─── File drop on URL input ───

function wireFileDrop() {
  const urlWrap = document.querySelector(".url-wrap");
  const urlInput = document.getElementById("url");
  const fileInput = document.getElementById("fileInput");
  const filePill = document.getElementById("filePill");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const fileClear = document.getElementById("fileClear");
  if (!urlWrap || !urlInput || !fileInput || !filePill) return;

  function formatBytes(n) {
    return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  function applyFile(file) {
    if (!file) return;
    const lower = file.name.toLowerCase();
    const supported = [".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"];
    if (!supported.some((ext) => lower.endsWith(ext))) {
      alert("Supported audio files: AAC, AIFF, FLAC, M4A, MP3, OGG, OPUS, and WAV.");
      return;
    }
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatBytes(file.size);
    filePill.classList.remove("hidden");
    urlWrap.classList.add("has-file");
    // Store file on the hidden input for job.js to pick up
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    urlInput.value = "";
    urlInput.removeAttribute("required");
  }

  function clearFile() {
    filePill.classList.add("hidden");
    urlWrap.classList.remove("has-file");
    fileInput.value = "";
    urlInput.setAttribute("required", "");
  }

  fileClear?.addEventListener("click", clearFile);

  urlWrap.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    urlWrap.classList.add("drag-over");
  });
  urlWrap.addEventListener("dragleave", (e) => {
    if (!urlWrap.contains(e.relatedTarget)) urlWrap.classList.remove("drag-over");
  });
  urlWrap.addEventListener("drop", (e) => {
    e.preventDefault();
    urlWrap.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) applyFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) applyFile(fileInput.files[0]);
  });
}

// ─── App shell controls ───

function wireAppShellControls() {
  document.getElementById("appMenuBtn")?.addEventListener("click", (e) => {
    e.stopPropagation();
    document.getElementById("catalogToggle")?.click();
  });

}

// ─── Keyboard shortcuts ───

document.addEventListener("keydown", (e) => {
  if (!multitrack) return;
  if (
    e.target instanceof HTMLInputElement
    || e.target instanceof HTMLSelectElement
    || e.target instanceof HTMLTextAreaElement
    || e.target?.isContentEditable
  ) return;
  if (e.code === "Space") {
    e.preventDefault();
    togglePlayPause();
  } else if (e.code === "BracketLeft") {
    e.preventDefault();
    nudgeTransport(-5);
  } else if (e.code === "BracketRight") {
    e.preventDefault();
    nudgeTransport(5);
  } else if (e.code === "KeyL") {
    e.preventDefault();
    loopBtn.click();
  } else if (e.code === "KeyI" && loopEnabled && multitrack) {
    e.preventDefault();
    setLoopStart(Math.min(getActiveTransportTime(), loopEnd - 0.5));
    updateLoopRegionVisual();
  } else if (e.code === "KeyO" && loopEnabled && multitrack) {
    e.preventDefault();
    setLoopEnd(Math.max(getActiveTransportTime(), loopStart + 0.5));
    updateLoopRegionVisual();
  }
});

// ─── External links ───

document.addEventListener("click", (e) => {
  const dl = e.target.closest("a.lane-dl, a.upmix-dl");
  if (dl?.href) {
    const openUrl = window.__TAURI__?.core?.invoke;
    if (openUrl) {
      e.preventDefault();
      openUrl("open_url", { url: dl.href });
    }
    return;
  }
  const anchor = e.target.closest('a[target="_blank"]');
  if (anchor?.href) {
    const openUrl = window.__TAURI__?.core?.invoke;
    if (openUrl) {
      e.preventDefault();
      openUrl("open_url", { url: anchor.href });
    }
  }
});

// ─── Global error logging ───

window.addEventListener("error", (e) => {
  console.error("[app:error]", e.message, "\n", e.filename, ":", e.lineno, "\n", e.error?.stack ?? "");
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("[app:unhandledrejection]", e.reason?.message ?? e.reason, "\n", e.reason?.stack ?? "");
});

// ─── Bootstrap ───

buildStripStems();
renderEmptyShell();
