# ComfyUI-WKW-StepPrinting 🎬🏮

A professional ComfyUI custom node pack that converts any input video (at any frame rate) into the iconic **Wong Kar-Wai** (*Chungking Express*, *Fallen Angels*, *Happy Together*) step-printed slow-motion aesthetic.

Built with **Deep Learning Optical Flow (torchvision RAFT)**, **VFX-grade forward-backward occlusion & disocclusion handling**, **continuous sub-pixel forward-splatted vector path integration**, **open shutter aperture smearing**, and **unified multi-frame global shutter exposure**.

---

## 📽️ The Cinematography Behind the Effect

In cinematography, Wong Kar-Wai and legendary DP Christopher Doyle achieved their signature "smeared time" look using a combination of two techniques: **undercranking** and **optical step-printing** (as demonstrated in [Austin Schmidt's cinematography analysis](https://www.austinschmidt.com/blog-/step-printing-wong-kar-wai)):

1. **Undercranking & Open Shutter**:
   - The camera shoots at an intentionally low frame rate (e.g., $6\,\text{fps}$, $8\,\text{fps}$, or $12\,\text{fps}$).
   - The shutter angle is set wide open ($180^\circ$ to $360^\circ$). At $6\,\text{fps}$ with a $360^\circ$ shutter, each exposure lasts $1/6\,\text{s}$ (4× longer than normal $24\,\text{fps}$ at $1/24\,\text{s}$).
   - Fast-moving subjects streak across the frame with heavy, fluid motion blur, while stationary backgrounds remain sharp.

2. **Optical Step-Printing**:
   - In post-production, each captured frame is duplicated (step-printed) onto a standard $24\,\text{fps}$ timeline (e.g., repeating each $6\,\text{fps}$ frame 4 times: `AAAA BBBB CCCC`).
   - This creates a hypnotic staccato cadence: time appears to stutter and slow down, with moving subjects trailing ghosted streaks of light.

3. **Continuous Forward Splatting & Occlusion Handling**:
   - Computes dense bidirectional optical flow (forward $I_t \to I_{t+1}$ and backward $I_{t+1} \to I_t$) using PyTorch RAFT.
   - Forward-splats pixels continuously across the active exposure window, avoiding discrete ghosting.
   - **Occlusion Z-Ordering**: Moving foreground objects receive higher Z-depth priority, preventing background elements from bleeding through solid foreground streaks.

---

## 📦 Nodes Included

### 1. 🎬 `WKW Step Printing (Optical Flow & Occlusion)`
The master step-printing engine:
- **`images`**: Input video frames (`IMAGE` batch `[B, H, W, C]`).
- **`input_fps`**: Original video framerate (e.g. `24.0`, `30.0`, `60.0`, or `96.0` from RIFE).
- **`target_capture_fps`**: Undercranking rate (e.g. `6.0`, `8.0`, `12.0`).
- **`shutter_timing`**:
  - `trailing (past trail only)` *(Default)*: Streaks extend strictly into the past behind moving subjects. Prevents "ghosts from the future" from appearing ahead of motion.
  - `centered`: Symmetrical shutter window.
  - `leading (future trail)`: Front-curtain sync.
- **`shutter_angle`**: Shutter opening angle:
  - `180.0°`: Standard film shutter.
  - `360.0°`: Full open shutter (classic WKW smear).
  - `540.0° - 720.0°`: Extended trailing light streaks.
- **`optical_flow_method`**:
  - `RAFT-Small (Deep Learning)`: Fast, accurate deep optical flow model.
  - `RAFT-Large (Deep Learning)`: High-fidelity deep flow for intricate fine motion.
  - `DIS (Fast OpenCV)`: Ultra-fast CPU/preview fallback.
  - `Farneback (OpenCV)`: Classical optical flow.
- **`flow_samples`**: Sub-pixel trajectory samples per frame (default: `16`).
- **`occlusion_aware`**: Enables forward-backward consistency check and Z-depth priority.
- **`occlusion_threshold`**: Pixel error sensitivity for occlusion masking (default: `1.5`).
- **`shutter_curve`**:
  - `trailing_decay`: Exponential decay backwards in time (leaves a glowing phosphor/neon trail behind moving objects).
  - `gaussian`: Natural optical shutter curve.
  - `box`: Uniform physical exposure.
  - `triangle`: Symmetric linear falloff.
- **`cadence_mode`**:
  - `step_print_repeat`: Repeats smeared frames (`AAAA BBBB CCCC`) to preserve original video duration & timeline FPS.
  - `decimate_to_target_fps`: Emits unique capture frames at low FPS (e.g. true $6\,\text{fps}$).
  - `slow_motion_stretch`: Extends each capture frame by `step_repeat_count`, generating true slow motion.
- **`device`**: `auto` (detects CUDA, MPS, CPU), `cuda`, `mps`, or `cpu`.

### 2. 👁️ `WKW Motion Vector Visualizer`
VFX inspection node:
- Outputs color-coded HSV motion vector fields and grayscale occlusion/disocclusion masks.

---

## ⚡ Installation

### Option 1: ComfyUI Manager
Search for `ComfyUI-WKW-StepPrinting` in the ComfyUI Manager and click **Install**.

### Option 2: Manual Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jnxmx/ComfyUI-WKW-StepPrinting.git
pip install -r ComfyUI-WKW-StepPrinting/requirements.txt
```

---

## 🚀 NVIDIA GPU Acceleration

- The deep learning optical flow engine uses native PyTorch (`torchvision.models.optical_flow`).
- It runs with full CUDA acceleration on NVIDIA RTX cards without needing to compile separate C++ extensions.
- Automatic fallback to OpenCV DIS optical flow is supported if running without CUDA.

---

## 📄 License

MIT License. Crafted for filmmakers, VFX artists, and AI video creators.
