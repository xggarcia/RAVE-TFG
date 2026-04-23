# Claude Code Configuration - RAVE-TFG

## Behavioral Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER save working files, text/mds, or tests to the root folder
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files

## File Organization

- NEVER save to root folder — use the directories below
- Use `/src` for source code files
- Use `/tests` for test files
- Use `/docs` for documentation and markdown files
- Use `/tools` for utility scripts

## Project Architecture

- Keep files under 350 lines (target 250)
- Ensure input validation at system boundaries
- This is a Python project — no npm/node tooling

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries

## TFG Writing Rules

These rules apply whenever writing or editing TFG document sections.

- Write first in Spanish (Catalan/Spanish) for review; translate to English only after approval
- Every factual claim must be backed by a verifiable external source (academic paper, official doc, etc.)
- Provide all citation links in the text and in a references list at the end of each section
- NEVER assume technical details about the project — if uncertain, ask before writing
- Do not describe what the code "could do"; describe only what it actually does (verify in source)
- Use present tense as primary verb tense; perfect tense is acceptable for completed work
- Do not use passive voice
- Each section must follow: Context/motivation → Objective → Methodology → Results/contributions
- Keep citations in the format: [N] Author(s). (Year). Title. Source. URL
- Approximately 250 words for the abstract; section length proportional to complexity


# RAVE-TFG — desktop app instructions for Claude Code

## Context

RAVE-TFG is a Python tool for training and using RAVE (Realtime Audio Variational autoEncoder) neural audio models. It currently runs as an interactive CLI menu. We are converting it into a **PySide6 desktop app** while keeping the exact same Python backend.

## Single source of truth for UI/UX

The complete UI/UX design lives in `./design-mock/` as an HTML+React prototype.

- Open `design-mock/RAVE-TFG.html` in a browser to navigate the full app as a live mockup.
- Every screen, form field, state (loading, error, success, empty), color, typography choice and custom component (knobs, VU meters, waveform, spectrogram, phase XY pad, loss chart) is defined there.
- **Do not improvise UI.** When in doubt, consult the mock. If the mock is ambiguous, ask the user before inventing.

Files worth reading in the mock:
- `styles/tokens.css` — color palette, type scale, spacing, radii. Port these to `app/ui/tokens.qss`.
- `components/primitives.jsx` — Knob, Slider, VU, Waveform, Spectrogram, LineChart, Progress, Toggle, Checkbox, RadioGroup. These become custom `QWidget` subclasses.
- `components/shell.jsx` — TitleBar, Sidebar, StatusRail, PageHeader, ContentArea layout. Becomes `MainWindow`.
- `components/screens-*.jsx` — one file per section. Match these page-for-page.

## Target stack

| Concern | Library |
|---|---|
| UI framework | **PySide6** (Qt 6) |
| Styling | Qt Style Sheets (QSS) + per-widget `QPainter` for custom components |
| Live plots | **pyqtgraph** (loss curve, spectrogram, latent trajectory) |
| Audio I/O | **sounddevice** (streaming GUI) — duplex streams, block size as low as 64 samples |
| Heavy work | `QThread` / `QProcess` / `QThreadPool` — never block the UI thread |
| RAVE core | existing Python modules — **wrap, do not rewrite** |
| Resampling | **torchaudio.transforms.Resample** (GPU-capable) |
| Chunking / silence removal | **librosa** |
| Loudness normalization | **pyloudnorm** (EBU R128) |
| Format conversion | **soundfile** (libsndfile wrapper — wav/flac/ogg); use `torchaudio.load` + `soundfile.write` directly |
| Packaging | **uv** + `pyproject.toml`; lock with `uv lock` → `uv.lock` |

### Library decisions — rationale & exclusions

- **sounddevice over pyaudio**: pyaudio is older and more platform-quirky; avoid unless sounddevice fails on a target platform.
- **pyqtgraph over matplotlib for live updates**: GPU-accelerated, designed for real-time data, much faster for VU/spectrogram/PhasePad.
- **QProcess over QThread for training**: isolates CUDA memory from the GUI process — a training crash won't kill the UI; process is cleanly killable. Parse stdout line-by-line with regex (no MLflow/W&B unless user explicitly requests experiment tracking).
- **pydub is banned**: shells out to ffmpeg, adds fragile subprocess dependency. Use `torchaudio.load` + `soundfile.write` instead.
- **No standalone binary (PyInstaller/briefcase)**: PyInstaller produces ~2 GB bundles with PyTorch and has limited PyTorch support in briefcase. Ship `pyproject.toml` + install script; target users (ML practitioners) can manage a Python 3.12 env.
- **sounddevice + QThread for streaming**: run the stream callback off-thread, push buffers to the GUI via Qt signals; pair with numpy ring buffers for the GUI → inference → output pipeline.

## Non-negotiable rules

1. **Do not rewrite the RAVE core logic.** The existing CLI modules (preprocess, train, export, stream, dataset utilities) must stay importable as libraries. The GUI is a thin wrapper that calls them in worker threads/processes and wires their outputs (progress, logs, metrics) to Qt signals.
2. **Keep the CLI working** throughout the port. The desktop app should share backend modules with the CLI, not replace them.
3. **UI thread stays responsive.** Any of: disk I/O over ~50 ms, model loading, training, preprocessing, audio inference — runs off-thread. Stream status via signals.
4. **Match the mock.** Field labels, placements, defaults, helper text, hint copy, state variants, color of running vs done vs error — all come from the mock.
5. **Custom widgets with QPainter first, external libs second.** Knob, VU, Waveform, PhasePad are `QWidget` subclasses drawing in `paintEvent`. Ask before adding a UI dependency beyond PySide6 + pyqtgraph.
6. **Ask before each phase.** Before starting a phase, paste the relevant mock screenshot and enumerate the fields/states/widgets you will implement. Wait for approval.

## Phase plan

Work strictly in order. Finish and ship a phase before starting the next. Each phase ends with a working binary the user can run.

### Phase 0 — Scaffold
- `pyproject.toml` with PySide6 + pyqtgraph deps
- `app/__main__.py` entrypoint
- `app/ui/tokens.qss` ported from `design-mock/styles/tokens.css` (colors, radii, fonts). Load with `QApplication.setStyleSheet`.
- `app/ui/main_window.py` — `MainWindow` = sidebar + content stack + status rail + titlebar
- `app/ui/pages/home.py` — dashboard page per `screens-home.jsx`, navigation tiles wired to `stack.setCurrentWidget`
- Only Home is real; other routes render a `QLabel` placeholder

Acceptance: app launches, sidebar items switch the visible page, Home matches the mock visually.

### Phase 1 — Simple forms
Pages that are just config → start command → report result:
- Preprocess
- Export
- Clean user data (with confirm dialog per `screens-data.jsx:CleanScreen`)

For each, build a form page whose **Start** button launches the corresponding CLI function in a `QThread`, shows a progress panel, and emits `finished(success, message)`.

### Phase 2 — Train (live progress)
The biggest phase. Implement:
- Train model page (form + extra-configs grid from `screens-train.jsx`)
- **Resume banner**: scan `~/runs/<model>/…` for the latest checkpoint on form change; show banner with *Resume* / *Start fresh*
- Run training in a `QProcess` (preferred over `QThread` — isolation, killable)
- Parse stdout for step / loss / it/s / val_loss; emit signals; update the six-stat strip
- `pyqtgraph.PlotWidget` for the loss chart (acid-green stroke, same shape as mock)
- Reconstruction samples panel: display the latest waveform + spectrogram from training callback outputs
- Log stream `QPlainTextEdit` with color per level (INFO/WARN/ERR) and autoscroll toggle
- **Error state** per `screens-states.jsx:TrainErrorScreen` — CUDA OOM parsing + suggested-fixes panel

### Phase 3 — Dataset wizard
- `screens-data.jsx:DatasetWizardScreen` split-view layout
- Seven sub-flows (`do_all`, `first`, `preview`, `final`, `normalize`, `merge`, `convert`)
- `preview` (Select from previews) is the hardest: a `QListView` with embedded waveform widgets, keyboard shortcuts `A` / `R` / `Space` / `← →`, sounddevice playback of the selected clip

### Phase 4 — Advanced training
- Train prior
- Phase-aware training (detect phase subfolders, drag-reorder list)
- Phase anchors (PCA projection view — use `pyqtgraph.ScatterPlotItem`)

### Phase 5 — Full workflow
- Run preprocess → train → export as a chained pipeline, each stage's status in the stepper view from `screens-train.jsx:WorkflowScreen`

### Phase 6 — Streaming GUI (hero)
Most complex. Budget ~1–2 weeks.
- Audio I/O via `sounddevice` duplex stream, block size 256, mono, 44.1 kHz
- **Inference worker**: separate `QThread` loading up to 4 TorchScript models, blending outputs by per-slot weights
- **Slot widgets**: four `SlotPanel`s with custom knobs (Gain/Temp/Smooth), embedded live `pyqtgraph.ImageView` spectrogram, per-slot VU meter
- **PhasePad**: custom `QWidget` with mouse tracking, paints 4-corner radial color field + cursor + trail; emits `xyChanged(float, float)`
- **Master strip**: live VU L/R, master/dry-wet knobs, latency readout, LIVE indicator
- **Prior model strip**: toggleable, feeds latent trajectory into inference worker
- Start/stop cleanly; handle xruns; show buffer underrun warnings in status rail

## Project layout

```
rave-tfg/
├── CLAUDE.md              ← this file
├── design-mock/           ← the HTML prototype (reference only, do not modify)
├── pyproject.toml
├── rave_core/             ← existing CLI logic — keep importable
│   ├── preprocess.py
│   ├── train.py
│   ├── export.py
│   ├── stream.py
│   └── dataset/…
├── app/                   ← new desktop app
│   ├── __main__.py
│   ├── ui/
│   │   ├── tokens.qss
│   │   ├── main_window.py
│   │   ├── shell/         ← sidebar, status rail, titlebar
│   │   ├── widgets/       ← Knob, VU, Waveform, PhasePad, …
│   │   └── pages/         ← one file per screen
│   ├── workers/           ← QThread/QProcess wrappers for each core op
│   └── models/             ← dataclasses + stateful models (Runs, Models, Devices)
└── tests/
```

## Design tokens → QSS mapping

Port these from `design-mock/styles/tokens.css`. Use `QPalette` for colors that QSS cannot style (e.g. selection), `QSS` for everything else.

- `--bg-0..5`, `--fg-0..4`, `--line-0..2` → QSS color variables (hardcode per widget class)
- `--acid`, `--amber`, `--magenta`, `--blue` → accent role classes (`QPushButton[role="primary"]`, etc.)
- `--mono`, `--sans` → `QFontDatabase.addApplicationFont` for JetBrains Mono + Inter, set via `QFont`
- Radii → `border-radius` in QSS
- Focus rings, hover states → QSS `:hover`, `:focus`

## Communication protocol between UI and workers

Use Qt signals. Every worker exposes at minimum:
```python
class TrainWorker(QObject):
    progress = Signal(dict)         # {step, loss, it_s, eta}
    log      = Signal(str, str)     # (level, message)
    sample   = Signal(object)       # {target_wav, recon_wav, target_spec, recon_spec}
    failed   = Signal(str, str)     # (short, traceback)
    finished = Signal(dict)         # summary
```

UI pages listen on these and update widgets. Never access worker state directly from the UI thread.

## When working

- Before starting any phase, read the relevant `components/screens-*.jsx` top-to-bottom and list the widgets/fields/states you will implement. Confirm with the user.
- Before adding any third-party Python dep beyond PySide6 + pyqtgraph + sounddevice + existing RAVE deps — ask.
- Keep PRs/commits phase-scoped. Don't mix scaffolding with feature work.
- Screenshot the app after each phase and diff against the mock. Flag any divergence and ask before resolving it.