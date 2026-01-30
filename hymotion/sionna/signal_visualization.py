"""
Signal Visualization Module for Wireless Sensing

Provides comprehensive visualization tools for analyzing
wireless channel signals from motion-based simulations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import os
from typing import List, Dict, Any, Optional, Tuple


class SignalVisualizer:
    """
    Comprehensive visualization for wireless sensing signals.

    Creates various plots to analyze how motion affects wireless channels.
    """

    def __init__(self, output_dir: str = "output"):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Style settings
        plt.style.use('default')
        self.cmap_hot = 'hot'
        self.cmap_cool = 'viridis'
        self.cmap_phase = 'twilight'

    def plot_single_frame_analysis(
        self,
        signals: Dict[str, np.ndarray],
        frame_idx: int,
        title_prefix: str = "",
    ) -> str:
        """
        Create detailed analysis plot for a single frame.

        Shows CIR, CFR, and path information in detail.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. CIR Magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        cir_mag = signals.get("cir_magnitude", np.zeros(128))
        cir_mag_db = 20 * np.log10(cir_mag + 1e-12)
        tap_indices = np.arange(len(cir_mag))
        ax1.stem(tap_indices, cir_mag_db, linefmt='b-', markerfmt='bo', basefmt='k-')
        ax1.set_xlabel('Delay Tap Index')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Channel Impulse Response (CIR)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(cir_mag))

        # 2. CIR Phase
        ax2 = fig.add_subplot(gs[0, 1])
        cir_phase = signals.get("cir_phase", np.zeros(128))
        ax2.plot(tap_indices, cir_phase, 'g-', linewidth=1.5)
        ax2.set_xlabel('Delay Tap Index')
        ax2.set_ylabel('Phase (rad)')
        ax2.set_title('CIR Phase')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-np.pi, np.pi)

        # 3. Path Power Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        path_powers_db = signals.get("path_powers_db", np.array([]))
        if len(path_powers_db) > 0:
            sorted_powers = np.sort(path_powers_db)[::-1][:20]  # Top 20 paths
            ax3.bar(range(len(sorted_powers)), sorted_powers, color='coral', alpha=0.7)
            ax3.set_xlabel('Path Rank')
            ax3.set_ylabel('Power (dB)')
            ax3.set_title(f'Top Path Powers (N={len(path_powers_db)})')
        ax3.grid(True, alpha=0.3)

        # 4. CFR Magnitude
        ax4 = fig.add_subplot(gs[1, 0])
        cfr_mag = signals.get("cfr_magnitude", np.zeros(64))
        cfr_mag_db = 20 * np.log10(cfr_mag + 1e-12)
        subcarriers = np.arange(len(cfr_mag))
        ax4.plot(subcarriers, cfr_mag_db, 'b-', linewidth=1.5)
        ax4.fill_between(subcarriers, cfr_mag_db, alpha=0.3)
        ax4.set_xlabel('Subcarrier Index')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_title('Channel Frequency Response (CFR)')
        ax4.grid(True, alpha=0.3)

        # 5. CFR Phase
        ax5 = fig.add_subplot(gs[1, 1])
        cfr_phase = signals.get("cfr_phase", np.zeros(64))
        ax5.plot(subcarriers, cfr_phase, 'm-', linewidth=1.5)
        ax5.set_xlabel('Subcarrier Index')
        ax5.set_ylabel('Phase (rad)')
        ax5.set_title('CFR Phase')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-np.pi, np.pi)

        # 6. CFR Complex Plane (constellation-like)
        ax6 = fig.add_subplot(gs[1, 2])
        cfr = signals.get("cfr", np.zeros(64, dtype=np.complex128))
        if np.any(cfr != 0):
            ax6.scatter(cfr.real, cfr.imag, c=subcarriers, cmap='rainbow', s=30, alpha=0.7)
            ax6.set_xlabel('Real')
            ax6.set_ylabel('Imaginary')
            ax6.set_title('CFR Complex Plane')
            ax6.axis('equal')
            ax6.grid(True, alpha=0.3)

        # 7. Doppler Spectrum
        ax7 = fig.add_subplot(gs[2, 0])
        doppler = signals.get("doppler_spectrum", np.zeros(64))
        doppler_bins = np.arange(len(doppler)) - len(doppler) // 2
        ax7.fill_between(doppler_bins, doppler, alpha=0.5, color='purple')
        ax7.plot(doppler_bins, doppler, 'purple', linewidth=1.5)
        ax7.set_xlabel('Doppler Bin')
        ax7.set_ylabel('Power')
        ax7.set_title('Doppler Spectrum')
        ax7.grid(True, alpha=0.3)

        # 8. Signal Statistics
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        stats_text = [
            f"Frame: {frame_idx}",
            f"RSSI: {signals.get('rssi_dbm', -100):.1f} dBm",
            f"Num Paths: {signals.get('num_paths', 0)}",
            f"Mean Delay: {signals.get('mean_delay', 0)*1e9:.2f} ns",
            f"RMS Delay Spread: {signals.get('rms_delay_spread', 0)*1e9:.2f} ns",
            f"Coherence BW: {signals.get('coherence_bandwidth', 0)/1e6:.2f} MHz",
            f"Doppler Shift: {signals.get('doppler_shift', 0):.2f} Hz",
        ]
        ax8.text(0.1, 0.9, '\n'.join(stats_text), transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax8.set_title('Signal Statistics')

        # 9. MIMO Channel Matrix Visualization (magnitude of first subcarrier)
        ax9 = fig.add_subplot(gs[2, 2])
        mimo_channel = signals.get("mimo_channel", np.zeros((4, 4, 64)))
        if mimo_channel.size > 0:
            H = mimo_channel[:, :, 0]  # First subcarrier
            im = ax9.imshow(20 * np.log10(np.abs(H) + 1e-12), cmap='hot', aspect='equal')
            ax9.set_xlabel('TX Antenna')
            ax9.set_ylabel('RX Antenna')
            ax9.set_title('MIMO Channel (SC 0)')
            plt.colorbar(im, ax=ax9, label='|H| (dB)')

        # Main title
        title = f'{title_prefix}Frame {frame_idx} - Detailed Signal Analysis'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Save
        output_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_temporal_evolution(
        self,
        signals_list: List[Dict[str, np.ndarray]],
        prompt: str = "",
    ) -> str:
        """
        Create comprehensive temporal evolution visualization.

        Shows how all signal representations change over time.
        """
        num_frames = len(signals_list)
        if num_frames == 0:
            return ""

        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25)

        frames = np.arange(num_frames)

        # 1. CIR Waterfall (time vs delay)
        ax1 = fig.add_subplot(gs[0, :])
        cir_matrix = np.array([20 * np.log10(s.get("cir_magnitude", np.zeros(128)) + 1e-12)
                              for s in signals_list])
        im1 = ax1.imshow(cir_matrix.T, aspect='auto', origin='lower', cmap='hot',
                        extent=[0, num_frames-1, 0, cir_matrix.shape[1]])
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Delay Tap')
        ax1.set_title('CIR Magnitude Evolution (Waterfall)')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')

        # 2. CFR Waterfall
        ax2 = fig.add_subplot(gs[1, :])
        cfr_matrix = np.array([20 * np.log10(s.get("cfr_magnitude", np.zeros(64)) + 1e-12)
                              for s in signals_list])
        im2 = ax2.imshow(cfr_matrix.T, aspect='auto', origin='lower', cmap='viridis',
                        extent=[0, num_frames-1, 0, cfr_matrix.shape[1]])
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Subcarrier')
        ax2.set_title('CFR Magnitude Evolution (CSI Spectrogram)')
        plt.colorbar(im2, ax=ax2, label='Magnitude (dB)')

        # 3. CFR Phase Evolution
        ax3 = fig.add_subplot(gs[2, :])
        phase_matrix = np.array([s.get("cfr_phase", np.zeros(64)) for s in signals_list])
        im3 = ax3.imshow(phase_matrix.T, aspect='auto', origin='lower', cmap='twilight',
                        extent=[0, num_frames-1, 0, phase_matrix.shape[1]],
                        vmin=-np.pi, vmax=np.pi)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Subcarrier')
        ax3.set_title('CFR Phase Evolution')
        plt.colorbar(im3, ax=ax3, label='Phase (rad)')

        # 4. Doppler Time-Frequency
        ax4 = fig.add_subplot(gs[3, :])
        doppler_matrix = np.array([s.get("doppler_spectrum", np.zeros(64)) for s in signals_list])
        if np.max(doppler_matrix) > 0:
            doppler_matrix = doppler_matrix / (np.max(doppler_matrix) + 1e-12)
        im4 = ax4.imshow(doppler_matrix.T, aspect='auto', origin='lower', cmap='plasma',
                        extent=[0, num_frames-1, -32, 31])
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Doppler Bin')
        ax4.set_title('Doppler Spectrogram')
        plt.colorbar(im4, ax=ax4, label='Normalized Power')

        # 5. Time series plots
        ax5 = fig.add_subplot(gs[4, 0])
        rssi = [s.get("rssi_dbm", -100) for s in signals_list]
        ax5.plot(frames, rssi, 'b-o', markersize=3, linewidth=1.5)
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('RSSI (dBm)')
        ax5.set_title('RSSI over Time')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[4, 1])
        num_paths = [s.get("num_paths", 0) for s in signals_list]
        ax6.fill_between(frames, num_paths, alpha=0.3, color='green')
        ax6.plot(frames, num_paths, 'g-', linewidth=1.5)
        ax6.set_xlabel('Frame')
        ax6.set_ylabel('Number of Paths')
        ax6.set_title('Multipath Count over Time')
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[5, 0])
        delays = [s.get("rms_delay_spread", 0) * 1e9 for s in signals_list]
        ax7.plot(frames, delays, 'r-s', markersize=3, linewidth=1.5)
        ax7.set_xlabel('Frame')
        ax7.set_ylabel('RMS Delay Spread (ns)')
        ax7.set_title('Delay Spread over Time')
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[5, 1])
        doppler_shifts = [s.get("doppler_shift", 0) for s in signals_list]
        ax8.plot(frames, doppler_shifts, 'm-^', markersize=3, linewidth=1.5)
        ax8.set_xlabel('Frame')
        ax8.set_ylabel('Doppler Shift (Hz)')
        ax8.set_title('Doppler Shift over Time')
        ax8.grid(True, alpha=0.3)

        # Main title
        title = f'Temporal Signal Evolution - "{prompt}"' if prompt else 'Temporal Signal Evolution'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        output_path = os.path.join(self.output_dir, "temporal_evolution.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def create_animated_visualization(
        self,
        signals_list: List[Dict[str, np.ndarray]],
        prompt: str = "",
        fps: int = 5,
    ) -> str:
        """
        Create animated GIF showing signal evolution.
        """
        try:
            from PIL import Image
        except ImportError:
            print("PIL not available for animation")
            return ""

        num_frames = len(signals_list)
        if num_frames == 0:
            return ""

        images = []

        for frame_idx, signals in enumerate(signals_list):
            # Create frame visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # CIR
            ax = axes[0, 0]
            cir_mag = signals.get("cir_magnitude", np.zeros(128))
            cir_mag_db = 20 * np.log10(cir_mag + 1e-12)
            ax.plot(cir_mag_db, 'b-', linewidth=1.5)
            ax.set_xlabel('Delay Tap')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('CIR')
            ax.set_ylim(-80, 0)
            ax.grid(True, alpha=0.3)

            # CFR
            ax = axes[0, 1]
            cfr_mag = signals.get("cfr_magnitude", np.zeros(64))
            cfr_mag_db = 20 * np.log10(cfr_mag + 1e-12)
            ax.plot(cfr_mag_db, 'g-', linewidth=1.5)
            ax.set_xlabel('Subcarrier')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('CFR')
            ax.set_ylim(-60, 0)
            ax.grid(True, alpha=0.3)

            # Doppler
            ax = axes[1, 0]
            doppler = signals.get("doppler_spectrum", np.zeros(64))
            doppler_bins = np.arange(len(doppler)) - len(doppler) // 2
            ax.fill_between(doppler_bins, doppler, alpha=0.5, color='purple')
            ax.set_xlabel('Doppler Bin')
            ax.set_ylabel('Power')
            ax.set_title('Doppler')
            ax.grid(True, alpha=0.3)

            # Stats
            ax = axes[1, 1]
            ax.axis('off')
            stats = [
                f"Frame: {frame_idx}/{num_frames-1}",
                f"RSSI: {signals.get('rssi_dbm', -100):.1f} dBm",
                f"Paths: {signals.get('num_paths', 0)}",
                f"Delay Spread: {signals.get('rms_delay_spread', 0)*1e9:.2f} ns",
            ]
            ax.text(0.5, 0.5, '\n'.join(stats), transform=ax.transAxes,
                   fontsize=14, ha='center', va='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            fig.suptitle(f'Frame {frame_idx} - "{prompt}"' if prompt else f'Frame {frame_idx}',
                        fontsize=12, fontweight='bold')

            plt.tight_layout()

            # Save to buffer
            frame_path = os.path.join(self.output_dir, f"_anim_frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            images.append(Image.open(frame_path))

        # Create GIF
        output_path = os.path.join(self.output_dir, "signal_animation.gif")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )

        # Cleanup temp files
        for frame_idx in range(num_frames):
            frame_path = os.path.join(self.output_dir, f"_anim_frame_{frame_idx:04d}.png")
            if os.path.exists(frame_path):
                os.remove(frame_path)

        return output_path

    def plot_representation_quality(
        self,
        signals_list: List[Dict[str, np.ndarray]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Analyze and visualize quality metrics for each representation.

        Shows which representations are most informative for motion detection.
        """
        num_frames = len(signals_list)
        if num_frames < 2:
            return ""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Compute quality metrics
        representations = {
            'RSSI': [s.get("rssi_dbm", -100) for s in signals_list],
            'CIR Energy': [np.sum(s.get("cir_magnitude", np.zeros(1))**2) for s in signals_list],
            'CFR Mean': [np.mean(s.get("cfr_magnitude", np.zeros(1))) for s in signals_list],
            'Doppler Peak': [np.max(s.get("doppler_spectrum", np.zeros(1))) for s in signals_list],
            'Phase Std': [np.std(s.get("cfr_phase", np.zeros(1))) for s in signals_list],
            'Path Count': [float(s.get("num_paths", 0)) for s in signals_list],
        }

        metrics = {}
        for name, values in representations.items():
            values = np.array(values)
            if len(values) > 1 and np.std(values) > 0:
                # Dynamic range
                metrics[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / (np.abs(np.mean(values)) + 1e-12),
                    'snr': np.abs(np.mean(values)) / (np.std(values) + 1e-12),
                    'range': np.max(values) - np.min(values),
                }
            else:
                metrics[name] = {'mean': 0, 'std': 0, 'cv': 0, 'snr': 0, 'range': 0}

        # 1. Time series comparison
        colors = plt.cm.Set2(np.linspace(0, 1, len(representations)))
        for ax_idx, ((name, values), color) in enumerate(zip(representations.items(), colors)):
            row, col = ax_idx // 3, ax_idx % 3
            ax = axes[row, col]

            # Normalize for comparison
            values = np.array(values)
            if np.std(values) > 0:
                values_norm = (values - np.mean(values)) / np.std(values)
            else:
                values_norm = values

            ax.plot(values_norm, color=color, linewidth=2, label=name)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'{name}\nCV: {metrics[name]["cv"]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(self.output_dir, "representation_quality.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path


def create_summary_report(
    signals_list: List[Dict[str, np.ndarray]],
    prompt: str,
    config: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Create a comprehensive text report summarizing the analysis.
    """
    report_lines = [
        "=" * 70,
        "WIRELESS SENSING ANALYSIS REPORT",
        "=" * 70,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"Prompt:          {prompt}",
        f"Frequency:       {config.get('frequency', 0)/1e9:.2f} GHz",
        f"Bandwidth:       {config.get('bandwidth', 0)/1e6:.0f} MHz",
        f"Subcarriers:     {config.get('num_subcarriers', 0)}",
        f"TX Position:     {config.get('tx_position', [])}",
        f"RX Position:     {config.get('rx_position', [])}",
        f"Total Frames:    {len(signals_list)}",
        "",
        "SIGNAL STATISTICS",
        "-" * 40,
    ]

    if signals_list:
        rssi_vals = [s.get("rssi_dbm", -100) for s in signals_list]
        paths_vals = [s.get("num_paths", 0) for s in signals_list]
        delay_vals = [s.get("rms_delay_spread", 0) * 1e9 for s in signals_list]

        report_lines.extend([
            f"RSSI:            {np.mean(rssi_vals):.1f} +/- {np.std(rssi_vals):.1f} dBm",
            f"                 Range: [{np.min(rssi_vals):.1f}, {np.max(rssi_vals):.1f}] dBm",
            f"Path Count:      {np.mean(paths_vals):.1f} +/- {np.std(paths_vals):.1f}",
            f"                 Range: [{np.min(paths_vals):.0f}, {np.max(paths_vals):.0f}]",
            f"RMS Delay:       {np.mean(delay_vals):.2f} +/- {np.std(delay_vals):.2f} ns",
            "",
        ])

    # Motion sensitivity analysis
    report_lines.extend([
        "MOTION SENSITIVITY ANALYSIS",
        "-" * 40,
    ])

    representations = {
        'RSSI': [s.get("rssi_dbm", -100) for s in signals_list],
        'CIR Energy': [np.sum(s.get("cir_magnitude", np.zeros(1))**2) for s in signals_list],
        'CFR Mean': [np.mean(s.get("cfr_magnitude", np.zeros(1))) for s in signals_list],
        'Doppler Peak': [np.max(s.get("doppler_spectrum", np.zeros(1))) for s in signals_list],
        'Phase Std': [np.std(s.get("cfr_phase", np.zeros(1))) for s in signals_list],
    }

    cv_scores = {}
    for name, values in representations.items():
        values = np.array(values)
        mean_val = np.abs(np.mean(values))
        if mean_val > 0 and np.std(values) > 0:
            cv = np.std(values) / mean_val
        else:
            cv = 0
        cv_scores[name] = cv

    # Sort by CV
    sorted_cv = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)

    for name, cv in sorted_cv:
        sensitivity = "HIGH" if cv > 0.3 else "MEDIUM" if cv > 0.1 else "LOW"
        report_lines.append(f"  {name:15s}: CV = {cv:.4f} [{sensitivity}]")

    report_lines.extend([
        "",
        "RECOMMENDATION",
        "-" * 40,
    ])

    if sorted_cv:
        best = sorted_cv[0]
        report_lines.append(f"Best representation for motion sensing: {best[0]}")
        report_lines.append(f"Coefficient of Variation: {best[1]:.4f}")
        report_lines.append("")

        if best[0] == "Doppler Peak":
            report_lines.append("Doppler-based sensing is excellent for:")
            report_lines.append("  - Detecting motion direction and velocity")
            report_lines.append("  - Activity recognition (walking, running, etc.)")
        elif best[0] == "Phase Std":
            report_lines.append("Phase-based sensing is excellent for:")
            report_lines.append("  - Fine-grained motion detection")
            report_lines.append("  - Breathing and vital signs")
            report_lines.append("  - Hand gesture recognition")
        elif best[0] == "CIR Energy":
            report_lines.append("CIR-based sensing is excellent for:")
            report_lines.append("  - Overall body motion tracking")
            report_lines.append("  - Room occupancy detection")
            report_lines.append("  - Large movement classification")
        elif best[0] == "CFR Mean":
            report_lines.append("CFR/CSI-based sensing is excellent for:")
            report_lines.append("  - WiFi-based sensing applications")
            report_lines.append("  - Human activity recognition")
            report_lines.append("  - Compatible with commercial WiFi CSI tools")
        elif best[0] == "RSSI":
            report_lines.append("RSSI-based sensing is good for:")
            report_lines.append("  - Simple presence detection")
            report_lines.append("  - Coarse motion detection")
            report_lines.append("  - Works with basic WiFi hardware")

    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    report_text = '\n'.join(report_lines)

    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_path
