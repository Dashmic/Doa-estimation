import argparse
import sys
import os
import io as _io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

M, N = 5, 6
SENSOR_POSITIONS = np.array([0, 5, 10, 15, 20, 25, 6, 12, 36, 42, 48, 54])
P = len(SENSOR_POSITIONS)
DOA_MIN, DOA_MAX, DOA_STEP = -60, 60, 1
DOA_GRID = np.arange(DOA_MIN, DOA_MAX + DOA_STEP, DOA_STEP)
NUM_CLASSES = len(DOA_GRID)


def steering_vector(theta_deg: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    return np.exp(1j * np.pi * SENSOR_POSITIONS * np.sin(theta_rad))


def steering_matrix(thetas_deg) -> np.ndarray:
    return np.column_stack([steering_vector(t) for t in thetas_deg])


def simulate_signal(thetas_deg, T: int = 32, snr_db: float = 10.0) -> np.ndarray:
    K = len(thetas_deg)
    A = steering_matrix(thetas_deg)
    sigma2 = 10 ** (-snr_db / 10)
    S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)
    N_ = np.sqrt(sigma2 / 2) * (np.random.randn(P, T) + 1j * np.random.randn(P, T))
    return A @ S + N_


def sample_covariance(X: np.ndarray) -> np.ndarray:
    T = X.shape[1]
    return (X @ X.conj().T) / T


def cov_to_tensor(R_hat: np.ndarray) -> torch.Tensor:
    inp = np.stack([R_hat.real, R_hat.imag], axis=0).astype(np.float32)
    return torch.from_numpy(inp).unsqueeze(0)


def music_spectrum(R_hat: np.ndarray, K_est: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(R_hat)
    Un = eigenvectors[:, :P - K_est]
    spectrum = np.zeros(NUM_CLASSES)
    for i, theta in enumerate(DOA_GRID):
        a = steering_vector(theta)
        proj = a.conj() @ Un @ Un.conj().T @ a
        spectrum[i] = 1.0 / (np.abs(proj) + 1e-12)
    spectrum /= spectrum.max()
    return spectrum


def build_resnext50(num_classes: int = 121) -> nn.Module:
    try:
        import torchvision.models as models
        model = models.resnext50_32x4d(weights=None)
        old_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        model.fc = nn.Linear(2048, num_classes)
        return model
    except Exception as e:
        print(f"[WARN] torchvision model build failed: {e}")
        print("[WARN] Using lightweight stub model.")
        return _stub_model(num_classes)


class _stub_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 12 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = build_resnext50()
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict):
            state_dict = state.get('model_state_dict', state.get('state_dict', state.get('model_state', state)))
        else:
            state_dict = state
        try:
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as e:
            print(f"[WARN] Could not load checkpoint strictly: {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        print("[WARN] Running with random weights.")
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model: nn.Module, R_hat: np.ndarray, device: torch.device,
                  threshold: float = 0.45) -> tuple:
    t = cov_to_tensor(R_hat).to(device)
    logits = model(t).squeeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()
    detected_idx = np.where(probs >= threshold)[0]
    detections = [(DOA_GRID[i], probs[i]) for i in detected_idx]
    return probs, detections


DARK_BG = '#0d1117'
PANEL_BG = '#161b22'
BLUE = '#58a6ff'
GREEN = '#56d364'
ORANGE = '#e3b341'
RED = '#f85149'
GRAY = '#8b949e'
WHITE = '#e6edf3'

PRESET_SCENARIOS = [
    {"label": "3 Sources (easy)", "angles": [-30, 10, 45], "snr": 10},
    {"label": "2 Sources (close)", "angles": [-5, 5], "snr": 10},
    {"label": "5 Sources", "angles": [-50, -20, 0, 25, 50], "snr": 10},
    {"label": "Low SNR (0 dB)", "angles": [-20, 15], "snr": 0},
    {"label": "Single source", "angles": [30], "snr": 10},
    {"label": "8 Sources", "angles": [-55, -35, -15, 0, 15, 30, 45, 55], "snr": 10},
]


class DOADemoApp:
    def __init__(self, model: nn.Module, device: torch.device, init_angles=None, init_snr=10.0):
        self.model = model
        self.device = device
        self.angles = list(init_angles) if init_angles else [-20, 10, 35]
        self.snr_db = init_snr
        self.T = 32
        self.threshold = 0.45
        self._scenario_idx = 0

        plt.rcParams.update({
            'figure.facecolor': DARK_BG,
            'axes.facecolor': PANEL_BG,
            'axes.edgecolor': GRAY,
            'axes.labelcolor': WHITE,
            'text.color': WHITE,
            'xtick.color': GRAY,
            'ytick.color': GRAY,
            'grid.color': '#30363d',
            'grid.alpha': 0.6,
            'font.family': 'monospace',
        })

        self.fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
        self.fig.canvas.manager.set_window_title(
            'DOA Estimation Demo — CNN (cov_t32) vs MUSIC | TCA M=5 N=6 P=12'
        )
        self._build_layout()
        self.update()

    def _build_layout(self):
        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            left=0.06, right=0.97,
            top=0.93, bottom=0.18,
            hspace=0.45, wspace=0.35,
        )

        self.ax_title = self.fig.add_subplot(gs[0, :])
        self.ax_title.axis('off')
        self.title_text = self.ax_title.text(
            0.5, 0.5,
            'Deep Learning DOA Estimation — TCA (M=5, N=6, P=12)',
            ha='center', va='center', fontsize=16, color=BLUE, fontweight='bold'
        )

        self.ax_cov_re = self.fig.add_subplot(gs[1, 0])
        self.ax_cov_im = self.fig.add_subplot(gs[1, 1])
        self.ax_array = self.fig.add_subplot(gs[1, 2])
        self.ax_cnn = self.fig.add_subplot(gs[2, :2])
        self.ax_music = self.fig.add_subplot(gs[2, 2])

        ax_snr = self.fig.add_axes([0.06, 0.09, 0.25, 0.025], facecolor=PANEL_BG)
        ax_thr = self.fig.add_axes([0.06, 0.05, 0.25, 0.025], facecolor=PANEL_BG)
        ax_angle = self.fig.add_axes([0.38, 0.07, 0.25, 0.03], facecolor=PANEL_BG)
        ax_btn_run = self.fig.add_axes([0.67, 0.06, 0.10, 0.05])
        ax_btn_prev = self.fig.add_axes([0.79, 0.06, 0.08, 0.05])
        ax_btn_next = self.fig.add_axes([0.89, 0.06, 0.08, 0.05])

        self.sl_snr = Slider(ax_snr, 'SNR (dB)', -5, 25, valinit=self.snr_db,
                             color=BLUE, valstep=1)
        self.sl_thr = Slider(ax_thr, 'Threshold', 0.10, 0.90, valinit=self.threshold,
                             color=GREEN, valstep=0.01)
        self.tb_angle = TextBox(ax_angle, 'Source angles (°, space-separated)',
                                initial=' '.join(str(a) for a in self.angles),
                                color=PANEL_BG, hovercolor='#21262d')
        self.tb_angle.label.set_color(WHITE)

        self.btn_run = Button(ax_btn_run, 'Run ▶', color='#238636', hovercolor='#2ea043')
        self.btn_prev = Button(ax_btn_prev, '◀ Preset', color='#1f6feb', hovercolor='#388bfd')
        self.btn_next = Button(ax_btn_next, 'Preset ▶', color='#1f6feb', hovercolor='#388bfd')

        self.btn_run.label.set_color(WHITE)
        self.btn_prev.label.set_color(WHITE)
        self.btn_next.label.set_color(WHITE)

        self.sl_snr.on_changed(self._on_snr)
        self.sl_thr.on_changed(self._on_thr)
        self.btn_run.on_clicked(self._on_run)
        self.btn_prev.on_clicked(self._on_prev_preset)
        self.btn_next.on_clicked(self._on_next_preset)

    def _on_snr(self, val):
        self.snr_db = val
        self.update()

    def _on_thr(self, val):
        self.threshold = val
        self.update()

    def _on_run(self, event):
        try:
            raw = self.tb_angle.text.strip()
            self.angles = [float(x) for x in raw.split()]
            self.update()
        except ValueError:
            self.ax_title.texts[0].set_text(
                '⚠ Invalid angle input — use space-separated numbers e.g. -20 10 35'
            )
            self.fig.canvas.draw_idle()

    def _on_prev_preset(self, event):
        self._scenario_idx = (self._scenario_idx - 1) % len(PRESET_SCENARIOS)
        self._load_preset()

    def _on_next_preset(self, event):
        self._scenario_idx = (self._scenario_idx + 1) % len(PRESET_SCENARIOS)
        self._load_preset()

    def _load_preset(self):
        s = PRESET_SCENARIOS[self._scenario_idx]
        self.angles = s['angles']
        self.snr_db = s['snr']
        self.sl_snr.set_val(s['snr'])
        self.tb_angle.set_val(' '.join(str(a) for a in s['angles']))
        self.update()

    def update(self):
        angles = self.angles
        K = len(angles)

        X = simulate_signal(angles, T=self.T, snr_db=self.snr_db)
        R_hat = sample_covariance(X)
        probs, detections = run_inference(self.model, R_hat, self.device, self.threshold)
        K_est = max(1, K)
        spectrum = music_spectrum(R_hat, K_est)

        for ax, data, label, cmap in [
            (self.ax_cov_re, R_hat.real, 'Re{R̂}  —  Real part', 'RdBu_r'),
            (self.ax_cov_im, R_hat.imag, 'Im{R̂}  —  Imaginary part', 'PiYG'),
        ]:
            ax.clear()
            vmax = np.abs(data).max()
            ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(label, color=WHITE, fontsize=10, pad=4)
            ax.set_xlabel('Sensor index', color=GRAY, fontsize=8)
            ax.set_ylabel('Sensor index', color=GRAY, fontsize=8)
            ax.tick_params(colors=GRAY, labelsize=7)

        self.ax_array.clear()
        self.ax_array.set_facecolor(PANEL_BG)
        self.ax_array.scatter(SENSOR_POSITIONS, np.zeros(P), s=80, color=BLUE, zorder=3)
        self.ax_array.set_xlim(-3, 57)
        self.ax_array.set_ylim(-1, 1)
        self.ax_array.set_yticks([])
        self.ax_array.set_title('TCA Physical Layout  (P=12)', color=WHITE, fontsize=10)
        self.ax_array.set_xlabel('Position (× λ/2)', color=GRAY, fontsize=8)
        self.ax_array.tick_params(colors=GRAY, labelsize=7)
        self.ax_array.grid(True, axis='x', alpha=0.3)
        for pos in SENSOR_POSITIONS:
            self.ax_array.axvline(pos, color=BLUE, alpha=0.2, linewidth=0.8)

        self.ax_cnn.clear()
        self.ax_cnn.set_facecolor(PANEL_BG)
        bar_colors = [GREEN if p >= self.threshold else '#30363d' for p in probs]
        self.ax_cnn.bar(DOA_GRID, probs, width=0.8, color=bar_colors, zorder=2)

        for a in angles:
            self.ax_cnn.axvline(a, color=RED, linestyle='--', linewidth=1.5,
                                zorder=4, label=f'GT: {a}°' if a == angles[0] else None)

        self.ax_cnn.axhline(self.threshold, color=ORANGE, linestyle=':', linewidth=1.2,
                            label=f'τ* = {self.threshold:.2f}', zorder=3)
        self.ax_cnn.set_xlim(DOA_MIN - 1, DOA_MAX + 1)
        self.ax_cnn.set_ylim(0, 1.05)
        self.ax_cnn.set_ylabel('Sigmoid probability', color=GRAY, fontsize=9)
        self.ax_cnn.set_title(
            f'CNN (cov_t32)  —  {len(detections)} detected  |  '
            f'SNR={self.snr_db:.0f} dB  |  K_true={K}',
            color=WHITE, fontsize=11
        )
        self.ax_cnn.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=WHITE, loc='upper right')
        self.ax_cnn.tick_params(colors=GRAY, labelsize=8)
        self.ax_cnn.grid(True, alpha=0.3)

        detected_angles = [f"{d[0]:.0f}° ({d[1]:.2f})" for d in detections]
        self.ax_cnn.set_xlabel(
            f'DOA (degrees)   |   Detections: {", ".join(detected_angles) if detected_angles else "none"}',
            color=GRAY, fontsize=9
        )

        self.ax_music.clear()
        self.ax_music.set_facecolor(PANEL_BG)
        self.ax_music.plot(DOA_GRID, spectrum, color=ORANGE, linewidth=1.5, zorder=2)
        self.ax_music.fill_between(DOA_GRID, spectrum, alpha=0.15, color=ORANGE)
        for a in angles:
            self.ax_music.axvline(a, color=RED, linestyle='--', linewidth=1.5, zorder=4)
        self.ax_music.set_xlim(DOA_MIN - 1, DOA_MAX + 1)
        self.ax_music.set_ylim(0, 1.1)
        self.ax_music.set_title(f'MUSIC Pseudospectrum  (K_est={K_est})', color=WHITE, fontsize=10)
        self.ax_music.set_xlabel('DOA (degrees)', color=GRAY, fontsize=9)
        self.ax_music.set_ylabel('Normalised power', color=GRAY, fontsize=9)
        self.ax_music.tick_params(colors=GRAY, labelsize=8)
        self.ax_music.grid(True, alpha=0.3)

        gt_str = ', '.join(f'{a}°' for a in angles)
        self.title_text.set_text(
            f'DOA Demo  |  Ground truth: [{gt_str}]  |  '
            f'SNR={self.snr_db:.0f} dB  |  T={self.T}  |  '
            f'CNN detections: {len(detections)}'
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive DOA Estimation Demo')
    parser.add_argument('--angles', nargs='+', type=float, default=[-20, 10, 35])
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(SCRIPT_DIR, 'checkpoints', 'cov_t32_best.pth'))
    parser.add_argument('--demo-mode', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cpu') if args.cpu else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Initial angles: {args.angles}  SNR: {args.snr} dB")

    model = load_model(args.checkpoint, device)

    if args.demo_mode:
        init_angles = PRESET_SCENARIOS[0]['angles']
        init_snr = PRESET_SCENARIOS[0]['snr']
    else:
        init_angles = args.angles
        init_snr = args.snr

    app = DOADemoApp(model, device, init_angles=init_angles, init_snr=init_snr)
    app.show()


if __name__ == '__main__':
    main()
