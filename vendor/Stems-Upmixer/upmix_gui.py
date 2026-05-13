# -*- coding: utf-8 -*-
"""Small Tkinter GUI wrapper for upmix_cli.py."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable


APP_DIR = Path(__file__).resolve().parent
CLI_PATH = APP_DIR / "upmix_cli.py"
CONFIG_PATH = APP_DIR / ".upmix_gui.json"

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
STEM_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif")

AUDIO_FILETYPES = (
    ("Audio files", "*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.aac *.ogg *.opus"),
    ("All files", "*.*"),
)
CHECKPOINT_FILETYPES = (
    ("Checkpoint files", "*.safetensors *.yaml *.yml *.ckpt *.pt *.pth"),
    ("All files", "*.*"),
)


def local_tool(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    candidate = APP_DIR / "bin" / f"{name}{suffix}"
    return str(candidate) if candidate.exists() else name


def default_stem_checkpoint() -> str:
    candidates = (
        APP_DIR / "weights" / "de_limiter_best.safetensors",
        APP_DIR / "weights" / "de_limiter_best.ckpt",
        APP_DIR
        / "weights"
        / "de_limiter_sgi_stem_sections_2"
        / "de_limiter_best.safetensors",
        APP_DIR / "weights" / "de_limiter_sgi_stem_sections_2" / "de_limiter_best.ckpt",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def display_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


class UpmixGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Upmixer GUI")
        self.minsize(980, 720)

        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self.command_refresh_job: str | None = None

        self.vars: dict[str, tk.Variable] = {}
        self.stem_vars: dict[str, tk.BooleanVar] = {}
        self._create_vars()
        self._load_config()
        self._build_ui()
        self._attach_traces()
        self.refresh_command()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_log_queue)

    def _create_vars(self) -> None:
        defaults: dict[str, str | bool] = {
            "job_dir": "",
            "stem_dir": "",
            "input_file": "",
            "output_dir": "",
            "title": "",
            "lfe_mode": "light",
            "analysis_backend": "auto",
            "silent_stem_filter": "auto",
            "mix_profile": "auto",
            "focus_stem_mode": "auto",
            "focus_stem": "vocals",
            "vocal_gain_db": "",
            "stem_gain_mode": "off",
            "stem_aware_eq": "auto",
            "bass_fold_support": "auto",
            "reference_guard": "auto",
            "style_reference": "",
            "style_reference_strength": "0.35",
            "temporal_mastering": "auto",
            "temporal_mastering_profile": "auto",
            "temporal_mastering_strength": "0.65",
            "temporal_feedback": "auto",
            "space_bed": "off",
            "space_bed_strength": "0.85",
            "brickwall_recovery": "off",
            "brickwall_recovery_strength": "0.75",
            "stem_de_limiter": "off",
            "stem_de_limiter_checkpoint": default_stem_checkpoint(),
            "stem_de_limiter_mix": "",
            "stem_de_limiter_keep_stems": False,
            "render_bed": True,
            "render_flac": True,
            "render_stereo": True,
            "render_apple_tv": True,
            "stereo_normalize": "linear",
            "stereo_loudness_i": "-14.0",
            "stereo_loudness_tp": "-1.0",
            "surround_normalize": "loudnorm",
            "surround_loudness_i": "-14.0",
            "surround_loudness_tp": "-1.0",
            "master_gain": "0.84",
            "flac_compression_level": "8",
            "apple_tv_bitrate": "768k",
            "ffmpeg": local_tool("ffmpeg"),
            "ffprobe": local_tool("ffprobe"),
        }
        for key, value in defaults.items():
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            else:
                self.vars[key] = tk.StringVar(value=value)
        for stem in STEMS:
            self.stem_vars[stem] = tk.BooleanVar(value=True)

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        for key, value in data.get("vars", {}).items():
            var = self.vars.get(key)
            if var is not None:
                var.set(value)
        for stem, value in data.get("stems", {}).items():
            var = self.stem_vars.get(stem)
            if var is not None:
                var.set(bool(value))

    def _save_config(self) -> None:
        data = {
            "vars": {key: var.get() for key, var in self.vars.items()},
            "stems": {stem: var.get() for stem, var in self.stem_vars.items()},
        }
        try:
            CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _build_ui(self) -> None:
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))

        self.basic_tab = ttk.Frame(notebook, padding=10)
        self.sound_tab = ttk.Frame(notebook, padding=10)
        self.output_tab = ttk.Frame(notebook, padding=10)
        self.advanced_tab = ttk.Frame(notebook, padding=10)
        self.log_tab = ttk.Frame(notebook, padding=10)

        notebook.add(self.basic_tab, text="基本")
        notebook.add(self.sound_tab, text="音作り")
        notebook.add(self.output_tab, text="出力")
        notebook.add(self.advanced_tab, text="詳細")
        notebook.add(self.log_tab, text="ログ")
        self.notebook = notebook

        self._build_basic_tab()
        self._build_sound_tab()
        self._build_output_tab()
        self._build_advanced_tab()
        self._build_log_tab()
        self._build_action_bar()

    def _build_basic_tab(self) -> None:
        self.basic_tab.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(self.basic_tab, text="入力", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.columnconfigure(1, weight=1)

        self._path_row(input_frame, 0, "WebUIジョブ", "job_dir", "dir")
        self._path_row(input_frame, 1, "Stemフォルダ", "stem_dir", "dir", post_command=self.detect_stems)
        self._path_row(input_frame, 2, "元音源", "input_file", "audio")
        self._path_row(input_frame, 3, "出力フォルダ", "output_dir", "dir")

        ttk.Label(input_frame, text="タイトル").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(input_frame, textvariable=self.vars["title"]).grid(row=4, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(input_frame, text="自動検出", command=self.auto_fill_from_job).grid(
            row=4, column=2, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        stem_frame = ttk.LabelFrame(self.basic_tab, text="使用Stem", padding=10)
        stem_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for index, stem in enumerate(STEMS):
            ttk.Checkbutton(stem_frame, text=stem, variable=self.stem_vars[stem]).grid(
                row=0, column=index, sticky="w", padx=(0, 18)
            )
        ttk.Button(stem_frame, text="フォルダから検出", command=self.detect_stems).grid(row=0, column=len(STEMS), sticky="e")

        summary_frame = ttk.LabelFrame(self.basic_tab, text="コマンド", padding=10)
        summary_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        self.command_preview = ScrolledText(summary_frame, height=8, wrap="word", state="disabled")
        self.command_preview.grid(row=0, column=0, sticky="nsew")

    def _build_sound_tab(self) -> None:
        self.sound_tab.columnconfigure(0, weight=1)
        self.sound_tab.columnconfigure(1, weight=1)

        placement = ttk.LabelFrame(self.sound_tab, text="配置", padding=10)
        placement.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        processing = ttk.LabelFrame(self.sound_tab, text="補正", padding=10)
        processing.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._combo_row(placement, 0, "LFE", "lfe_mode", ("light", "normal", "off"))
        self._combo_row(placement, 1, "Mix profile", "mix_profile", ("auto", "front51"))
        self._combo_row(placement, 2, "解析", "analysis_backend", ("auto", "numpy", "ffmpeg"))
        self._combo_row(placement, 3, "無音Stem", "silent_stem_filter", ("auto", "off"))
        self._combo_row(placement, 4, "Focus", "focus_stem_mode", ("auto", "manual", "off"))
        self._combo_row(placement, 5, "Focus stem", "focus_stem", STEMS)
        self._entry_row(placement, 6, "Vocal gain dB", "vocal_gain_db")
        self._combo_row(placement, 7, "Stem gain", "stem_gain_mode", ("off", "spatial", "harman"))
        self._combo_row(placement, 8, "Stem EQ", "stem_aware_eq", ("auto", "off"))
        self._combo_row(placement, 9, "Bass fold", "bass_fold_support", ("auto", "off"))

        self._combo_row(processing, 0, "Reference", "reference_guard", ("auto", "off", "original"))
        self._path_row(processing, 1, "Style ref", "style_reference", "audio")
        self._entry_row(processing, 2, "Style strength", "style_reference_strength")
        self._combo_row(processing, 3, "Temporal", "temporal_mastering", ("auto", "off"))
        self._combo_row(
            processing,
            4,
            "Temporal profile",
            "temporal_mastering_profile",
            ("auto", "conservative", "natural", "open"),
        )
        self._entry_row(processing, 5, "Temporal strength", "temporal_mastering_strength")
        self._combo_row(processing, 6, "Feedback", "temporal_feedback", ("auto", "off"))
        self._combo_row(processing, 7, "Space bed", "space_bed", ("auto", "off"))
        self._entry_row(processing, 8, "Space strength", "space_bed_strength")
        self._combo_row(
            processing,
            9,
            "Brickwall",
            "brickwall_recovery",
            ("off", "auto", "conservative", "natural", "open"),
        )
        self._entry_row(processing, 10, "Brickwall strength", "brickwall_recovery_strength")

        delim = ttk.LabelFrame(self.sound_tab, text="Stem de-limiter", padding=10)
        delim.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        delim.columnconfigure(1, weight=1)
        self._combo_row(delim, 0, "Mode", "stem_de_limiter", ("off", "auto", "safe", "remaster", "repair", "sections2"))
        self._path_row(delim, 1, "Checkpoint", "stem_de_limiter_checkpoint", "checkpoint")
        self._entry_row(delim, 2, "Mix", "stem_de_limiter_mix")
        ttk.Checkbutton(
            delim,
            text="処理済みStemを残す",
            variable=self.vars["stem_de_limiter_keep_stems"],
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))

    def _build_output_tab(self) -> None:
        self.output_tab.columnconfigure(0, weight=1)
        self.output_tab.columnconfigure(1, weight=1)

        formats = ttk.LabelFrame(self.output_tab, text="形式", padding=10)
        formats.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Checkbutton(formats, text="7.1.4 WAVを残す", variable=self.vars["render_bed"]).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(formats, text="5.1 FLAC", variable=self.vars["render_flac"]).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Checkbutton(formats, text="Stereo FLAC", variable=self.vars["render_stereo"]).grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(formats, text="Apple TV MP4", variable=self.vars["render_apple_tv"]).grid(
            row=3, column=0, sticky="w", pady=2
        )

        loudness = ttk.LabelFrame(self.output_tab, text="ラウドネス", padding=10)
        loudness.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._combo_row(loudness, 0, "Stereo norm", "stereo_normalize", ("linear", "loudnorm", "off"))
        self._entry_row(loudness, 1, "Stereo LUFS", "stereo_loudness_i")
        self._entry_row(loudness, 2, "Stereo TP", "stereo_loudness_tp")
        self._combo_row(loudness, 3, "5.1 norm", "surround_normalize", ("loudnorm", "off"))
        self._entry_row(loudness, 4, "5.1 LUFS", "surround_loudness_i")
        self._entry_row(loudness, 5, "5.1 TP", "surround_loudness_tp")

        render = ttk.LabelFrame(self.output_tab, text="レンダー", padding=10)
        render.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        render.columnconfigure(1, weight=1)
        self._entry_row(render, 0, "Master gain", "master_gain")
        self._entry_row(render, 1, "FLAC level", "flac_compression_level")
        self._entry_row(render, 2, "Apple bitrate", "apple_tv_bitrate")
        self._path_row(render, 3, "ffmpeg", "ffmpeg", "exe")
        self._path_row(render, 4, "ffprobe", "ffprobe", "exe")

    def _build_advanced_tab(self) -> None:
        self.advanced_tab.columnconfigure(0, weight=1)
        self.advanced_tab.rowconfigure(1, weight=1)
        ttk.Label(self.advanced_tab, text="追加引数").grid(row=0, column=0, sticky="w")
        self.extra_args = ScrolledText(self.advanced_tab, height=8, wrap="word")
        self.extra_args.grid(row=1, column=0, sticky="nsew", pady=(6, 10))
        self.extra_args.bind("<KeyRelease>", lambda _event: self.queue_command_refresh())

        buttons = ttk.Frame(self.advanced_tab)
        buttons.grid(row=2, column=0, sticky="ew")
        ttk.Button(buttons, text="コマンド更新", command=self.refresh_command).pack(side="left")
        ttk.Button(buttons, text="コマンドをコピー", command=self.copy_command).pack(side="left", padx=(8, 0))

    def _build_log_tab(self) -> None:
        self.log_tab.columnconfigure(0, weight=1)
        self.log_tab.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(self.log_tab, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _build_action_bar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        bar.grid(row=1, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="待機中")
        ttk.Label(bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        ttk.Button(bar, text="出力フォルダを開く", command=self.open_output_dir).grid(row=0, column=1, padx=(8, 0))
        self.start_button = ttk.Button(bar, text="開始", command=self.start_process)
        self.start_button.grid(row=0, column=2, padx=(8, 0))
        self.stop_button = ttk.Button(bar, text="停止", command=self.stop_process, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=(8, 0))

    def _path_row(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        key: str,
        kind: str,
        post_command: Callable[[], None] | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 0))
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", pady=(8 if row else 0, 0))
        ttk.Button(
            parent,
            text="参照",
            command=lambda: self.browse_path(key, kind, post_command=post_command),
        ).grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=(8 if row else 0, 0))
        ttk.Button(parent, text="消去", command=lambda: self.vars[key].set("")).grid(
            row=row,
            column=3,
            sticky="ew",
            padx=(6, 0),
            pady=(8 if row else 0, 0),
        )

    def _combo_row(self, parent: tk.Misc, row: int, label: str, key: str, values: tuple[str, ...]) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 0))
        ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly").grid(
            row=row,
            column=1,
            sticky="ew",
            pady=(8 if row else 0, 0),
        )

    def _entry_row(self, parent: tk.Misc, row: int, label: str, key: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 0))
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=(8 if row else 0, 0),
        )

    def _attach_traces(self) -> None:
        for var in [*self.vars.values(), *self.stem_vars.values()]:
            var.trace_add("write", lambda *_args: self.queue_command_refresh())

    def browse_path(self, key: str, kind: str, post_command: Callable[[], None] | None = None) -> None:
        current = str(self.vars[key].get())
        current_path = Path(current) if current else None
        if current_path is not None and current_path.is_file():
            initial = str(current_path.parent)
        elif current_path is not None and current_path.exists():
            initial = str(current_path)
        else:
            initial = str(APP_DIR)
        selected = ""
        if kind == "dir":
            selected = filedialog.askdirectory(initialdir=initial)
        elif kind == "audio":
            selected = filedialog.askopenfilename(initialdir=initial, filetypes=AUDIO_FILETYPES)
        elif kind == "checkpoint":
            selected = filedialog.askopenfilename(initialdir=initial, filetypes=CHECKPOINT_FILETYPES)
        else:
            selected = filedialog.askopenfilename(initialdir=initial, filetypes=(("Executables", "*.exe"), ("All files", "*.*")))
        if selected:
            self.vars[key].set(selected)
            if post_command is not None:
                post_command()

    def auto_fill_from_job(self) -> None:
        job_dir = Path(str(self.vars["job_dir"].get()).strip())
        if not job_dir.exists():
            messagebox.showwarning("Upmixer GUI", "WebUIジョブフォルダが見つかりません。")
            return

        output_root = job_dir / "output"
        if output_root.exists():
            children = sorted(path for path in output_root.iterdir() if path.is_dir())
            if len(children) == 1:
                self.vars["stem_dir"].set(str(children[0]))
            elif len(children) > 1:
                self._append_log("Stem候補が複数あります。Stemフォルダを手動で選んでください。\n")

        input_root = job_dir / "input"
        if input_root.exists():
            candidates = sorted(path for path in input_root.iterdir() if path.is_file())
            if candidates:
                self.vars["input_file"].set(str(candidates[0]))

        self.detect_stems()
        self.refresh_command()

    def detect_stems(self) -> None:
        stem_dir_text = str(self.vars["stem_dir"].get()).strip()
        if not stem_dir_text:
            return
        stem_dir = Path(stem_dir_text)
        if not stem_dir.exists():
            messagebox.showwarning("Upmixer GUI", "Stemフォルダが見つかりません。")
            return

        found: list[str] = []
        for stem in STEMS:
            exists = any((stem_dir / f"{stem}{ext}").exists() for ext in STEM_EXTENSIONS)
            self.stem_vars[stem].set(exists)
            if exists:
                found.append(stem)
        self._append_log(f"検出Stem: {', '.join(found) if found else 'なし'}\n")

    def selected_stems(self) -> list[str]:
        return [stem for stem in STEMS if self.stem_vars[stem].get()]

    def queue_command_refresh(self) -> None:
        if self.command_refresh_job is not None:
            self.after_cancel(self.command_refresh_job)
        self.command_refresh_job = self.after(250, self.refresh_command)

    def refresh_command(self) -> None:
        self.command_refresh_job = None
        try:
            command = self.build_command(validate=False)
            text = display_command(command)
        except ValueError as exc:
            text = f"入力エラー: {exc}"
        self.command_preview.configure(state="normal")
        self.command_preview.delete("1.0", "end")
        self.command_preview.insert("1.0", text)
        self.command_preview.configure(state="disabled")

    def copy_command(self) -> None:
        try:
            command = self.build_command(validate=False)
        except ValueError as exc:
            messagebox.showerror("Upmixer GUI", str(exc))
            return
        self.clipboard_clear()
        self.clipboard_append(display_command(command))
        self.status_var.set("コマンドをコピーしました")

    def build_command(self, *, validate: bool) -> list[str]:
        errors: list[str] = []
        command = [sys.executable, str(CLI_PATH)]

        def text(key: str) -> str:
            return str(self.vars[key].get()).strip()

        def add_text(option: str, key: str) -> None:
            value = text(key)
            if value:
                command.extend([option, value])

        def add_float(option: str, key: str, *, optional: bool = False) -> None:
            value = text(key)
            if not value and optional:
                return
            if not value:
                errors.append(f"{key} が空です")
                return
            try:
                float(value)
            except ValueError:
                errors.append(f"{key} は数値で入力してください")
                return
            command.extend([option, value])

        def add_int(option: str, key: str) -> None:
            value = text(key)
            try:
                int(value)
            except ValueError:
                errors.append(f"{key} は整数で入力してください")
                return
            command.extend([option, value])

        if not text("job_dir") and not text("stem_dir"):
            errors.append("WebUIジョブまたはStemフォルダを選んでください")
        for option, key in (
            ("--job-dir", "job_dir"),
            ("--stem-dir", "stem_dir"),
            ("--input", "input_file"),
            ("--output-dir", "output_dir"),
            ("--title", "title"),
        ):
            add_text(option, key)

        stems = self.selected_stems()
        if not stems:
            errors.append("使用Stemを1つ以上選んでください")
        command.extend(["--stems", ",".join(stems)])

        if not (
            bool(self.vars["render_bed"].get())
            or bool(self.vars["render_flac"].get())
            or bool(self.vars["render_stereo"].get())
            or bool(self.vars["render_apple_tv"].get())
        ):
            errors.append("出力形式を1つ以上選んでください")

        simple_options = (
            ("--lfe-mode", "lfe_mode"),
            ("--analysis-backend", "analysis_backend"),
            ("--silent-stem-filter", "silent_stem_filter"),
            ("--mix-profile", "mix_profile"),
            ("--focus-stem-mode", "focus_stem_mode"),
            ("--focus-stem", "focus_stem"),
            ("--stem-gain-mode", "stem_gain_mode"),
            ("--stem-aware-eq", "stem_aware_eq"),
            ("--bass-fold-support", "bass_fold_support"),
            ("--reference-guard", "reference_guard"),
            ("--temporal-mastering", "temporal_mastering"),
            ("--temporal-mastering-profile", "temporal_mastering_profile"),
            ("--temporal-feedback", "temporal_feedback"),
            ("--space-bed", "space_bed"),
            ("--brickwall-recovery", "brickwall_recovery"),
            ("--stem-de-limiter", "stem_de_limiter"),
            ("--stereo-normalize", "stereo_normalize"),
            ("--surround-normalize", "surround_normalize"),
            ("--apple-tv-bitrate", "apple_tv_bitrate"),
            ("--ffmpeg", "ffmpeg"),
            ("--ffprobe", "ffprobe"),
        )
        for option, key in simple_options:
            add_text(option, key)

        add_float("--vocal-gain-db", "vocal_gain_db", optional=True)
        add_text("--style-reference", "style_reference")
        add_float("--style-reference-strength", "style_reference_strength")
        add_float("--temporal-mastering-strength", "temporal_mastering_strength")
        add_float("--space-bed-strength", "space_bed_strength")
        add_float("--brickwall-recovery-strength", "brickwall_recovery_strength")

        stem_de_limiter = text("stem_de_limiter")
        checkpoint = text("stem_de_limiter_checkpoint")
        if stem_de_limiter not in {"off", "safe"}:
            if not checkpoint:
                checkpoint = default_stem_checkpoint()
                if checkpoint:
                    self.vars["stem_de_limiter_checkpoint"].set(checkpoint)
            if not checkpoint:
                errors.append("Stem de-limiter checkpointを選んでください")
            add_text("--stem-de-limiter-checkpoint", "stem_de_limiter_checkpoint")
        add_float("--stem-de-limiter-mix", "stem_de_limiter_mix", optional=True)
        if bool(self.vars["stem_de_limiter_keep_stems"].get()):
            command.append("--stem-de-limiter-keep-stems")

        add_float("--stereo-loudness-i", "stereo_loudness_i")
        add_float("--stereo-loudness-tp", "stereo_loudness_tp")
        add_float("--surround-loudness-i", "surround_loudness_i")
        add_float("--surround-loudness-tp", "surround_loudness_tp")
        add_float("--master-gain", "master_gain")
        add_int("--flac-compression-level", "flac_compression_level")

        if not bool(self.vars["render_flac"].get()):
            command.append("--skip-flac")
        if not bool(self.vars["render_stereo"].get()):
            command.append("--skip-stereo")
        if not bool(self.vars["render_apple_tv"].get()):
            command.append("--skip-apple-tv")
        if not bool(self.vars["render_bed"].get()):
            command.append("--skip-bed")

        extra = self.extra_args.get("1.0", "end").strip() if hasattr(self, "extra_args") else ""
        if extra:
            try:
                command.extend(shlex.split(extra))
            except ValueError as exc:
                errors.append(f"追加引数を解析できません: {exc}")

        if validate:
            self._validate_paths(errors)
        if errors:
            raise ValueError("\n".join(errors))
        return command

    def _validate_paths(self, errors: list[str]) -> None:
        for key, label in (
            ("job_dir", "WebUIジョブ"),
            ("stem_dir", "Stemフォルダ"),
            ("input_file", "元音源"),
            ("style_reference", "Style ref"),
        ):
            value = str(self.vars[key].get()).strip()
            if value and not Path(value).exists():
                errors.append(f"{label} が見つかりません: {value}")

        stem_de_limiter = str(self.vars["stem_de_limiter"].get()).strip()
        checkpoint = str(self.vars["stem_de_limiter_checkpoint"].get()).strip()
        if stem_de_limiter not in {"off", "safe"} and checkpoint and not Path(checkpoint).exists():
            errors.append(f"Stem de-limiter checkpoint が見つかりません: {checkpoint}")

        for key, label in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")):
            value = str(self.vars[key].get()).strip()
            if value and any(sep in value for sep in ("/", "\\")) and not Path(value).exists():
                errors.append(f"{label} が見つかりません: {value}")

    def start_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        try:
            command = self.build_command(validate=True)
        except ValueError as exc:
            messagebox.showerror("Upmixer GUI", str(exc))
            return

        self._save_config()
        self._clear_log()
        self._append_log(display_command(command) + "\n\n")
        self.status_var.set("実行中")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.notebook.select(self.log_tab)

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.status_var.set("起動失敗")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showerror("Upmixer GUI", str(exc))
            return

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None:
            return
        assert process.stdout is not None
        for line in process.stdout:
            self.log_queue.put(("line", line))
        return_code = process.wait()
        self.log_queue.put(("done", return_code))

    def stop_process(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.status_var.set("停止中")
        self._append_log("\n停止要求を送信しました。\n")
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()

    def _drain_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._append_log(str(payload))
            elif kind == "done":
                code = int(payload)
                if code == 0:
                    self.status_var.set("完了")
                else:
                    self.status_var.set(f"終了コード {code}")
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self._append_log(f"\n終了コード: {code}\n")
        self.after(100, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def inferred_output_dir(self) -> Path | None:
        output_dir = str(self.vars["output_dir"].get()).strip()
        if output_dir:
            return Path(output_dir)
        job_dir = str(self.vars["job_dir"].get()).strip()
        if job_dir:
            return Path(job_dir) / "upmix"
        stem_dir = str(self.vars["stem_dir"].get()).strip()
        if stem_dir:
            return Path(stem_dir) / "upmix"
        return None

    def open_output_dir(self) -> None:
        output_dir = self.inferred_output_dir()
        if output_dir is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(output_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Upmixer GUI", "処理中です。停止して閉じますか？"):
                return
            self.stop_process()
        self._save_config()
        self.destroy()


def main() -> None:
    app = UpmixGui()
    app.mainloop()


if __name__ == "__main__":
    main()
