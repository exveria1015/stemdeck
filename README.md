<div align="center">

![StemDeck](stemdeck.png)

# StemDeck

**Free, local stem separation. No account. No upload. No subscription.**

<div align="center">
  <a href="https://github.com/thcp/stemdeck/stargazers"><img src="https://img.shields.io/github/stars/thcp/stemdeck?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/thcp/stemdeck/releases"><img src="https://img.shields.io/github/downloads/thcp/stemdeck/total?style=flat-square&color=52c65f" alt="Total Downloads"></a>
  <a href="https://github.com/thcp/stemdeck/releases/latest"><img src="https://img.shields.io/github/v/release/thcp/stemdeck?style=flat-square" alt="Latest Release"></a>
  <a href="https://github.com/thcp/stemdeck/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thcp/stemdeck?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/CI-Woodpecker-4D9DE0?style=flat-square&logo=woodpecker-ci&logoColor=white" alt="CI: Woodpecker">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-0078D6?style=flat-square&logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/Powered_by-Residual%20Allocator-FF6B35?style=flat-square" alt="Powered by Residual Allocator">
</div>

</div>

<br>

Drop an audio file (MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, AIFF), or paste a YouTube URL. StemDeck splits the audio into up to six stems (vocals, drums, bass, guitar, piano, other) and plays them back in a DAW-style multitrack mixer. Mute, solo, mix, zoom the waveform, loop a region, and download individual stems or a custom mix. Everything runs on your own machine.

> **What is this?** StemDeck is a stem separation tool, not a downloader. Its primary use case is processing audio files you already own — drag a supported audio file onto the import bar and go. YouTube URL support is provided as a convenience for content you have the right to process. StemDeck does not store, cache, or redistribute any downloaded content. All processing happens locally on your machine and nothing leaves it.

> StemDeck is a free, open alternative to cloud stem-splitters like Moises and LALAL.AI. No account, no quota, no upload, no subscription. If you mainly want stems for personal study and prefer to keep things local and free, StemDeck should be enough. If you need the polish, the mobile app, or the extra musician tooling, the commercial products are a better fit.

If you find StemDeck useful, consider [buying the maker a coffee](https://buymeacoffee.com/tha.les); these donations are being used to random acts of kindness toward others 

---

## Features

**6-stem separation** via a local Residual Allocator checkout, with auto-detection of the best Torch device (CUDA on NVIDIA, MPS on Apple Silicon, CPU fallback). Demucs remains available as a fallback backend.

**YouTube and local file import.** Paste a YouTube URL or drop a supported audio file directly onto the import bar.

**DAW-style waveform editor** with min/max sample rendering across all stems, shared normalization, zoom in/out/Fit, loop drag on the ruler, gold playhead overlay, and stem-aligned lanes.

**Stem subset extraction.** Click stem chips to choose which stems to keep. Clicking from "all selected" snaps to "only this one"; subsequent clicks add or remove.

**"Original" backing track.** When you pick a subset, a 7th lane contains the complement (full song minus selected stems), perfect for A/B reference without doubling.

**Downloadable selected mix.** A single `mix.wav` of just your selected stems, summed by the Rust native audio helper when available, with FFmpeg kept as a fallback.

**Optional surround upmix.** Enable **Upmix** before processing to run the vendored `vendor/Stems-Upmixer` pipeline after separation. The normal stem player still opens as usual, and the Information panel exposes the rendered 7.1.4 WAV, 5.1 FLAC, stereo FLAC, and Apple TV MP4 outputs when available.

**Per-stem mixer** with volume fader, mute, solo, and "monitor" (solo-only) per stem. State syncs between the preview mixer and the stems sidebar.

**Live VU meters** per stem. Post-gain RMS via Web Audio analysers with peak hold and slow falloff.

**Song analysis** including BPM (librosa beat tracker), key, scale, and confidence (Albrecht-Shanahan profiles), integrated LUFS (BS.1770), and sample peak in dBFS.

**Cancellable jobs.** Cancel mid-pipeline and the runner terminates the active subprocess immediately, deletes the partial job dir, and returns to ready.

**Library panel** with folder-based track organisation, drag-and-drop, search, and trash.

---

## Honest Comparison

StemDeck is not trying to compete with commercial stem-separation products. It covers the core use case well and stops there. This table exists so you can make an informed choice rather than discover the gaps after the fact.

| | StemDeck | Moises / LALAL.AI / similar |
|---|---|---|
| **Price** | Free, forever | Freemium; credits or subscription required for regular use |
| **Hosting** | Runs entirely on your machine | Cloud; audio must be uploaded to their servers |
| **Account / login** | None | Required |
| **Internet required** | Only for YouTube download once the local model files are present | Always; no offline use |
| **Privacy** | Audio never leaves your machine | Audio is uploaded and processed on third-party servers |
| **Data retention** | You control it; delete anytime | Governed by their privacy policy and retention period |
| **Stem model** | Local Residual Allocator over BS-RoFormer stems | Proprietary models, regularly updated, generally higher quality |
| **Stem count** | 6 (vocals, drums, bass, guitar, piano, other) | Up to 10 depending on service and plan |
| **Input formats** | YouTube URL, MP3, WAV | MP3, WAV, FLAC, M4A, and more depending on service |
| **Processing speed** | Depends on your hardware; fast with a GPU, slow on CPU only | Fast regardless of your hardware (runs on their servers) |
| **Batch processing** | One job at a time | Yes, on paid plans |
| **Mobile app** | No | iOS and Android |
| **Extra features** | No (no pitch shift, chord detection, lyrics, click track, BPM tap) | Yes, varies by product |
| **Polish** | Functional, hobby-grade UI | Polished, production-grade apps |
| **Source code** | Open source, forkable, self-hostable | Closed source |

If you need speed, quality, mobile access, or the extra musician tooling, the commercial products are worth the money. If you want stems for personal study, prefer to keep audio private, or just want something that runs locally with no strings attached, StemDeck is enough.

---

## Download

Pre-built portable zips are attached to each [GitHub Release](https://github.com/thcp/stemdeck/releases). No installer needed; extract and run.

| Zip | GPU | Approx. size |
|---|---|---|
| `StemDeck-Windows-x64.zip` | CPU only | ~700 MB |
| `StemDeck-Windows-x64.NVIDIA.zip` | NVIDIA CUDA | ~1.6 GB |

Extract the zip anywhere, run `StemDeck.exe`. On first launch the app verifies the bundled Python runtime and downloads FFmpeg. Subsequent launches skip this and start in seconds. Everything is self-contained; no Python or system dependencies required.

---

## Technologies

StemDeck is built on **[Python 3.10+](https://python.org)** managed via **[uv](https://github.com/astral-sh/uv)**, with a **[FastAPI](https://fastapi.tiangolo.com)** backend serving REST and Server-Sent Events. Stem separation uses the vendored **Residual Allocator** checkout at `vendor/Residual-Allocator` by default, or Demucs when `STEMDECK_SEPARATOR=demucs` is set. Optional surround rendering uses the vendored **Stems-Upmixer** checkout at `vendor/Stems-Upmixer`. YouTube audio is fetched via **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**; WAV stem summing uses the Rust native audio helper, while FFmpeg remains available for codec/container routes such as non-WAV input preparation and Apple TV MP4. BPM detection and key analysis run on **[librosa](https://librosa.org)**; loudness measurement uses **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)** (ITU-R BS.1770). The Windows desktop shell is **[Tauri v2](https://tauri.app)** (Rust/WebView2). The frontend is vanilla JS with the Web Audio API, no framework and no build step; waveforms are rendered on `<canvas>` using min/max sample rendering.

*Thanks to the creators and maintainers of all the open-source libraries that make StemDeck possible.*

---

## Build from Source

*(macOS / Linux / Windows with Python 3.10+)*

### Prerequisites

Python 3.10 or newer, [uv](https://github.com/astral-sh/uv), and `ffmpeg` on your PATH for non-WAV import/transcode and Apple TV-style outputs. A Rust toolchain is optional in source builds; when present, StemDeck auto-builds `native_audio` for WAV stem summing and otherwise falls back to FFmpeg. By default StemDeck uses `vendor/Residual-Allocator` with its `weights/BS-Rofo-SW-Fixed.ckpt` and `weights/residual_allocator.safetensors` files present.

### macOS / Linux (one-shot)

```sh
git clone https://github.com/thcp/stemdeck stemdeck && cd stemdeck
./run.sh setup     # installs ffmpeg + uv, runs uv sync
./run.sh start
```

Open <http://localhost:8000>.

`setup` uses Homebrew on macOS and `apt-get` on Debian/Ubuntu. For other Linux distros, install `ffmpeg` and [uv](https://github.com/astral-sh/uv) manually, then run `uv sync` followed by `./run.sh start`.

### Manual (any platform)

```sh
git clone https://github.com/thcp/stemdeck stemdeck && cd stemdeck
uv sync
uv run uvicorn app.main:app --reload
```

### Docker

```sh
docker compose -f build/docker-compose.yml up --build
```

Stems land in `./jobs/` on the host. Residual Allocator model files are read from the vendored checkout or the configured override. Note: no GPU passthrough on macOS Docker.

### `run.sh` control script

```sh
./run.sh setup      # one-shot: install ffmpeg + uv, then uv sync
./run.sh start      # boots uvicorn in the background
./run.sh stop       # graceful shutdown
./run.sh restart    # stop + start
./run.sh status     # is it running?
```

---

## How to Use

1. On the import bar, click stem chips to choose which stems to extract (defaults to all 6). Enable **Upmix** when you also want surround exports after separation.
2. Paste a YouTube URL **or** drop a supported audio file, then click **Process**.
3. Wait through `Uploading...` / `Downloading...` → `Analyzing...` → `Separating...` → `Mixing tracks...`; with **Upmix** enabled, the job continues through `Rendering surround upmix...`.
4. When done, the studio dashboard appears. If you picked a subset, the first lane is **Original** (full song minus your selection); the rest are your isolated stems.
5. Mix: **Play/Pause/Stop** controls the master transport. **M** mutes a stem, **S** solos it (additive; multiple solos stay audible), **Monitor** solos only that stem and clears others. The volume fader moves 1:1 with drag; double-click resets to 0 dB; `Shift+wheel` gives coarse adjustment and plain wheel gives fine. The **Reset**, **Mute**, and **Solo** toolbar buttons act on all stems at once.
6. Drag on the ruler to define a loop region; click `Loop` to enable. Use `+` / `-` / `Fit` or `Ctrl/Cmd+wheel` to zoom.
7. **Download Mix** in the footer gives you a WAV of your selected stems summed together.

**Keyboard shortcuts:** `Space` play/pause · `[` seek -5s · `]` seek +5s · `L` loop · `I` loop in · `O` loop out

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `STEMDECK_SEPARATOR` | `residual_allocator` | Separator backend: `residual_allocator` or `demucs`. |
| `STEMDECK_RESIDUAL_ALLOCATOR_DIR` | `vendor/Residual-Allocator` | Local Residual Allocator checkout. Relative paths are resolved from the StemDeck root. |
| `STEMDECK_RESIDUAL_ALLOCATOR_DEVICE` | auto | Force Torch device for Residual Allocator: `cuda`, `mps`, or `cpu`. |
| `STEMDECK_RESIDUAL_ALLOCATOR_ARGS` | empty | Extra CLI arguments appended to `infer.py`. |
| `STEMDECK_DEMUCS_DEVICE` | auto | Force Torch device: `cuda`, `mps`, or `cpu`. |
| `STEMDECK_DEMUCS_MODEL` | `htdemucs_6s` | Demucs model name. |
| `STEMDECK_UPMIXER_DIR` | `vendor/Stems-Upmixer` | Local Stems-Upmixer checkout used when the Upmix toggle is enabled. Relative paths are resolved from the StemDeck root. |
| `STEMDECK_UPMIXER_SCRIPT` | `<upmixer dir>/upmix_cli.py` | Stems-Upmixer CLI entrypoint. |
| `STEMDECK_UPMIXER_ARGS` | empty | Extra CLI arguments appended to `upmix_cli.py`, for example `--skip-bed --skip-apple-tv`. |
| `STEMDECK_NATIVE_AUDIO_BIN` | auto-build from `native_audio` | Optional prebuilt `stemdeck-native-audio` helper used for WAV `original.wav` / `mix.wav` summing and native 7.1.4 bed rendering before falling back to FFmpeg. |
| `STEMS_UPMIXER_RENDER_714_BACKEND` | `auto` | 7.1.4 bed renderer: `auto` uses Rust for stem pan/EQ/LFE, temporal automation, space-bed delay/allpass, and focus/priority sidechain ducking, then falls back to FFmpeg on helper failures; `native` fails instead of falling back; `ffmpeg` forces the legacy renderer. |
| `STEMDECK_JOBS_DIR` | `./jobs` | Where job directories land. |
| `STEMDECK_MAX_DURATION_SEC` | `1200` | Reject audio longer than this (seconds). |
| `STEMDECK_JOB_TTL_SECONDS` | `86400` | How long to keep job dirs on disk. |
| `STEMDECK_MAX_PENDING_JOBS` | `3` | Max queued jobs before returning 503. |

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/jobs` | JSON `{url, stems?, upmix?}` or multipart `file + stems + upmix` → `{job_id}` |
| GET | `/api/jobs/{id}` | Job state snapshot |
| GET | `/api/jobs/{id}/events` | SSE stream of job state |
| POST | `/api/jobs/{id}/cancel` | Terminate active subprocess and cancel job |
| GET | `/api/jobs/{id}/stems/{name}.wav` | Stream/download a single stem (range requests) |
| GET | `/api/jobs/{id}/upmix/{filename}` | Stream/download an upmix output produced for that job |
| DELETE | `/api/jobs/{id}` | Remove job dir from disk (terminal jobs only) |

---

## Troubleshooting

**`ffmpeg: command not found`:** install ffmpeg and restart with `./run.sh restart`.

**`WARNING: [youtube] No supported JavaScript runtime`:** install deno (`brew install deno` on macOS) and restart. Downloads still work without it but may pick suboptimal formats.

**Residual Allocator is not configured:** verify `vendor/Residual-Allocator/infer.py`, `weights/BS-Rofo-SW-Fixed.ckpt`, and `weights/residual_allocator.safetensors` exist, or set `STEMDECK_RESIDUAL_ALLOCATOR_DIR`. Relative overrides are resolved from the StemDeck root.

**Separation runs on CPU only:** check the startup log for `separator_device=mps` or `separator_device=cuda`. If you see `cpu`, your torch install may be CPU-only.

**Page reloaded mid-job:** the job keeps running server-side. Wait for it to finish, then resubmit.

**`./run.sh: Permission denied`:** run `chmod +x run.sh`.

---

## Layout on Disk

```
jobs/<job_id>/
├── stems/
    ├── vocals.wav      # the 6 separator stems (always present)
    ├── drums.wav
    ├── bass.wav
    ├── guitar.wav
    ├── piano.wav
    ├── other.wav
    ├── original.wav    # sum of un-selected stems (subset only)
    └── mix.wav         # native Rust sum of selected stems, FFmpeg fallback (subset only)
└── upmix/
    ├── *_7.1.4_*.wav
    ├── *_upmix5.1_*.flac
    ├── *_stereo_*.flac
    └── *_apple_tv.mp4
```

Job state is in-memory. Restart the server and the job list resets, but files persist on disk. Old dirs are swept automatically (TTL 24 h, configurable).

---

## Disclaimer

StemDeck is a local audio stem separation tool intended for personal study, research, and experimentation. It is not a downloading service. It does not store, cache, or redistribute any audio content. All processing runs on the user's own machine and no audio is transmitted anywhere.

YouTube URL support is provided via [yt-dlp](https://github.com/yt-dlp/yt-dlp) as a convenience. Automated downloading may violate YouTube's Terms of Service. You, the user, are solely responsible for ensuring you have the right to process any audio you submit, complying with the terms of service of any site you download from, and respecting the copyright of the material you work with.

You are also responsible for following the licenses of the underlying tools this project depends on (yt-dlp, Residual Allocator, Demucs, FFmpeg, PyTorch, and others listed in `pyproject.toml`).

The author(s) of StemDeck provide this software "as is", without warranty of any kind, and accept no responsibility or liability for how it is used.

---

## Contributing

Issues, feature suggestions, and pull requests are welcome. See open issues for what's planned.

---

## Star History

<a href="https://www.star-history.com/?type=date&repos=thcp%2Fstemdeck">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=thcp/stemdeck&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=thcp/stemdeck&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=thcp/stemdeck&type=date&legend=top-left" />
 </picture>
</a>
