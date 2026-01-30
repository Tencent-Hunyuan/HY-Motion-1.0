#!/usr/bin/env python3
"""
Motion to Wireless Sensing Example

This example demonstrates:
1. Generating human motion from text prompt using HY-Motion
2. Simulating wireless channel using Sionna ray tracing
3. Analyzing frame-by-frame signal variations at RX antenna
4. Comparing different signal representations (CIR, CFR, Doppler, etc.)
5. Determining optimal signal representation for motion recognition

Goal: Understand how human motion affects wireless signals,
      preparing for future work on:
      - LLM prompt -> Motion -> Wireless signal
      - Wireless signal -> LLM prompt (inverse problem)

Usage:
    python examples/motion_wireless_sensing.py --prompt "A person waving hand"
    python examples/motion_wireless_sensing.py --prompt "걷는 사람" --frequency 5.8e9
"""

import argparse
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_signal_analysis_plot(
    signals_list: list,
    output_path: str,
    prompt: str,
):
    """
    Create comprehensive signal analysis visualization.

    Shows frame-by-frame evolution of all signal representations.
    """
    num_frames = len(signals_list)
    if num_frames == 0:
        print("No signals to visualize")
        return

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. RSSI over time
    ax_rssi = fig.add_subplot(gs[0, 0])
    rssi_values = [s.get("rssi_dbm", -100) for s in signals_list]
    frames = list(range(num_frames))
    ax_rssi.plot(frames, rssi_values, 'b-o', linewidth=2, markersize=4)
    ax_rssi.set_xlabel('Frame')
    ax_rssi.set_ylabel('RSSI (dBm)')
    ax_rssi.set_title('Received Signal Strength over Time')
    ax_rssi.grid(True, alpha=0.3)

    # 2. Number of paths over time
    ax_paths = fig.add_subplot(gs[0, 1])
    num_paths = [s.get("num_paths", 0) for s in signals_list]
    ax_paths.bar(frames, num_paths, color='green', alpha=0.7)
    ax_paths.set_xlabel('Frame')
    ax_paths.set_ylabel('Number of Paths')
    ax_paths.set_title('Multipath Count over Time')
    ax_paths.grid(True, alpha=0.3)

    # 3. RMS Delay Spread over time
    ax_delay = fig.add_subplot(gs[0, 2])
    rms_delays = [s.get("rms_delay_spread", 0) * 1e9 for s in signals_list]  # in ns
    ax_delay.plot(frames, rms_delays, 'r-s', linewidth=2, markersize=4)
    ax_delay.set_xlabel('Frame')
    ax_delay.set_ylabel('RMS Delay Spread (ns)')
    ax_delay.set_title('Delay Spread over Time')
    ax_delay.grid(True, alpha=0.3)

    # 4. CIR Magnitude Heatmap (frame vs delay)
    ax_cir = fig.add_subplot(gs[1, :])
    cir_matrix = np.array([s.get("cir_magnitude", np.zeros(128))[:128] for s in signals_list])
    # Normalize for better visualization
    cir_matrix_db = 20 * np.log10(cir_matrix + 1e-12)
    im_cir = ax_cir.imshow(
        cir_matrix_db.T,
        aspect='auto',
        origin='lower',
        cmap='hot',
        extent=[0, num_frames-1, 0, cir_matrix.shape[1]],
    )
    ax_cir.set_xlabel('Frame')
    ax_cir.set_ylabel('Delay Tap Index')
    ax_cir.set_title('Channel Impulse Response (CIR) Evolution')
    plt.colorbar(im_cir, ax=ax_cir, label='Magnitude (dB)')

    # 5. CFR Magnitude Heatmap (frame vs subcarrier)
    ax_cfr = fig.add_subplot(gs[2, :])
    cfr_matrix = np.array([s.get("cfr_magnitude", np.zeros(64))[:64] for s in signals_list])
    cfr_matrix_db = 20 * np.log10(cfr_matrix + 1e-12)
    im_cfr = ax_cfr.imshow(
        cfr_matrix_db.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, num_frames-1, 0, cfr_matrix.shape[1]],
    )
    ax_cfr.set_xlabel('Frame')
    ax_cfr.set_ylabel('Subcarrier Index')
    ax_cfr.set_title('Channel Frequency Response (CFR/CSI) Evolution')
    plt.colorbar(im_cfr, ax=ax_cfr, label='Magnitude (dB)')

    # 6. Doppler Spectrum Heatmap
    ax_doppler = fig.add_subplot(gs[3, 0:2])
    doppler_matrix = np.array([s.get("doppler_spectrum", np.zeros(64))[:64] for s in signals_list])
    if np.max(doppler_matrix) > 0:
        doppler_matrix_norm = doppler_matrix / (np.max(doppler_matrix) + 1e-12)
    else:
        doppler_matrix_norm = doppler_matrix
    im_doppler = ax_doppler.imshow(
        doppler_matrix_norm.T,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        extent=[0, num_frames-1, -32, 31],
    )
    ax_doppler.set_xlabel('Frame')
    ax_doppler.set_ylabel('Doppler Bin')
    ax_doppler.set_title('Doppler Spectrum Evolution')
    plt.colorbar(im_doppler, ax=ax_doppler, label='Normalized Power')

    # 7. Phase Variation over time (first few subcarriers)
    ax_phase = fig.add_subplot(gs[3, 2])
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for sc_idx in range(min(5, signals_list[0].get("cfr_phase", np.zeros(64)).shape[0] if hasattr(signals_list[0].get("cfr_phase", np.zeros(64)), "shape") else 64)):
        phase_values = [s.get("cfr_phase", np.zeros(64))[sc_idx] if len(s.get("cfr_phase", [])) > sc_idx else 0 for s in signals_list]
        ax_phase.plot(frames, phase_values, color=colors[sc_idx], label=f'SC {sc_idx}', alpha=0.7)
    ax_phase.set_xlabel('Frame')
    ax_phase.set_ylabel('Phase (rad)')
    ax_phase.set_title('Phase Evolution (per Subcarrier)')
    ax_phase.legend(loc='upper right', fontsize=8)
    ax_phase.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(f'Wireless Sensing Signal Analysis\nPrompt: "{prompt}"', fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Signal analysis plot saved to: {output_path}")


def create_representation_comparison_plot(
    comparison_results: dict,
    signals_list: list,
    output_path: str,
):
    """
    Create visualization comparing different signal representations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    temporal_variance = comparison_results.get("temporal_variance", {})
    ranked = comparison_results.get("ranked_representations", [])

    # 1. Bar plot of temporal variance
    ax = axes[0, 0]
    if temporal_variance:
        names = list(temporal_variance.keys())
        values = list(temporal_variance.values())
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
        bars = ax.bar(names, values, color=colors)
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Motion Sensitivity by Representation')
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2-6. Time series for each representation
    num_frames = len(signals_list)
    frames = list(range(num_frames))

    representations = {
        (0, 1): ("rssi", "rssi_dbm", "RSSI (dBm)"),
        (0, 2): ("cir_energy", None, "CIR Energy"),
        (1, 0): ("cfr_mean", None, "CFR Mean"),
        (1, 1): ("doppler_peak", None, "Doppler Peak"),
        (1, 2): ("phase_std", None, "Phase Std Dev"),
    }

    for (row, col), (name, key, title) in representations.items():
        ax = axes[row, col]

        if key:
            values = [s.get(key, 0) for s in signals_list]
        elif name == "cir_energy":
            values = [np.sum(s.get("cir_magnitude", np.zeros(1))**2) for s in signals_list]
        elif name == "cfr_mean":
            values = [np.mean(s.get("cfr_magnitude", np.zeros(1))) for s in signals_list]
        elif name == "doppler_peak":
            values = [np.max(s.get("doppler_spectrum", np.zeros(1))) for s in signals_list]
        elif name == "phase_std":
            values = [np.std(s.get("cfr_phase", np.zeros(1))) for s in signals_list]
        else:
            values = [0] * num_frames

        ax.plot(frames, values, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Frame')
        ax.set_ylabel(title)
        ax.set_title(f'{title} over Time')
        ax.grid(True, alpha=0.3)

        # Add CV annotation
        cv = temporal_variance.get(name, 0)
        ax.annotate(f'CV: {cv:.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Representation comparison plot saved to: {output_path}")


def print_analysis_summary(comparison_results: dict, signals_list: list):
    """Print summary of signal analysis."""
    print("\n" + "=" * 60)
    print("SIGNAL REPRESENTATION ANALYSIS SUMMARY")
    print("=" * 60)

    temporal_variance = comparison_results.get("temporal_variance", {})
    ranked = comparison_results.get("ranked_representations", [])

    print("\nMotion Sensitivity Ranking (higher CV = more sensitive):")
    print("-" * 40)
    for i, (name, cv) in enumerate(ranked, 1):
        sensitivity = "HIGH" if cv > 0.3 else "MEDIUM" if cv > 0.1 else "LOW"
        print(f"  {i}. {name:20s}: CV = {cv:.4f}  [{sensitivity}]")

    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR WIRELESS SENSING")
    print("=" * 60)

    if ranked:
        best = ranked[0]
        print(f"\nBest representation for motion detection: {best[0].upper()}")
        print(f"  - Coefficient of Variation: {best[1]:.4f}")

        if best[0] == "doppler_peak":
            print("  - Doppler is excellent for detecting velocity/movement")
            print("  - Recommended for: walking, running, arm movements")
        elif best[0] == "phase_std":
            print("  - Phase is sensitive to small position changes")
            print("  - Recommended for: breathing, hand gestures")
        elif best[0] == "cir_energy":
            print("  - CIR captures multipath changes from body movement")
            print("  - Recommended for: overall body motion, room occupancy")
        elif best[0] == "cfr_mean":
            print("  - CFR provides frequency-domain channel state")
            print("  - Recommended for: WiFi CSI-based sensing")
        elif best[0] == "rssi":
            print("  - RSSI is simplest but least detailed")
            print("  - Recommended for: coarse presence detection")

    # Signal statistics
    print("\n" + "=" * 60)
    print("SIGNAL STATISTICS")
    print("=" * 60)

    if signals_list:
        rssi_values = [s.get("rssi_dbm", -100) for s in signals_list]
        num_paths = [s.get("num_paths", 0) for s in signals_list]
        delay_spreads = [s.get("rms_delay_spread", 0) * 1e9 for s in signals_list]

        print(f"\nRSSI:           {np.mean(rssi_values):.1f} +/- {np.std(rssi_values):.1f} dBm")
        print(f"Path Count:     {np.mean(num_paths):.1f} +/- {np.std(num_paths):.1f}")
        print(f"Delay Spread:   {np.mean(delay_spreads):.2f} +/- {np.std(delay_spreads):.2f} ns")
        print(f"Total Frames:   {len(signals_list)}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Motion to Wireless Sensing Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python motion_wireless_sensing.py --prompt "A person walking forward"
  python motion_wireless_sensing.py --prompt "손을 흔드는 사람" --frequency 5.8e9
  python motion_wireless_sensing.py --prompt "A person doing jumping jacks" --duration 5.0

Signal Representations:
  - CIR: Channel Impulse Response (time domain)
  - CFR: Channel Frequency Response (frequency domain, like WiFi CSI)
  - Doppler: Frequency shift from motion
  - RSSI: Received Signal Strength
  - Phase: Phase changes across subcarriers
        """
    )

    parser.add_argument("--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0",
                       help="Path to HY-Motion model")
    parser.add_argument("--prompt", type=str, default="A person walking forward",
                       help="Text prompt for motion generation")
    parser.add_argument("--duration", type=float, default=3.0,
                       help="Motion duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                       help="Classifier-free guidance scale")

    # RF parameters
    parser.add_argument("--frequency", type=float, default=3.5e9,
                       help="Carrier frequency in Hz (default: 3.5 GHz for 5G)")
    parser.add_argument("--bandwidth", type=float, default=100e6,
                       help="System bandwidth in Hz")
    parser.add_argument("--num_subcarriers", type=int, default=64,
                       help="Number of OFDM subcarriers")

    # TX/RX positions
    parser.add_argument("--tx", type=str, default="0,0,3",
                       help="Transmitter position (x,y,z) in meters")
    parser.add_argument("--rx", type=str, default="5,5,1.5",
                       help="Receiver position (x,y,z) in meters")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/wireless_sensing",
                       help="Output directory for results")
    parser.add_argument("--save_dataset", action="store_true",
                       help="Save data in ML-ready format")

    args = parser.parse_args()

    # Parse positions
    tx_position = [float(x) for x in args.tx.split(",")]
    rx_position = [float(x) for x in args.rx.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MOTION TO WIRELESS SENSING ANALYSIS")
    print("=" * 60)
    print(f"Prompt:      {args.prompt}")
    print(f"Duration:    {args.duration}s")
    print(f"Frequency:   {args.frequency/1e9:.2f} GHz")
    print(f"Bandwidth:   {args.bandwidth/1e6:.0f} MHz")
    print(f"Subcarriers: {args.num_subcarriers}")
    print(f"TX Position: {tx_position}")
    print(f"RX Position: {rx_position}")
    print(f"Output:      {args.output_dir}")
    print("=" * 60)

    # Import modules
    from hymotion.sionna.ray_visualization import SionnaRayVisualizer, RayTracingConfig
    from hymotion.sionna.wireless_sensing import (
        WirelessSensingProcessor,
        WirelessSensingConfig,
        MotionWirelessDataset,
        compare_signal_representations,
        compute_signal_features,
    )
    from hymotion.utils.t2m_runtime import T2MRuntime

    # Initialize motion runtime
    config_path = os.path.join(args.model_path, "config.yml")
    ckpt_path = os.path.join(args.model_path, "latest.ckpt")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return 1

    print("\n[1/5] Loading HY-Motion model...")
    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        disable_prompt_engineering=True,
    )

    # Configure ray tracing
    ray_config = RayTracingConfig(
        frequency=args.frequency,
        tx_position=tx_position,
        rx_position=rx_position,
    )

    # Configure wireless sensing
    sensing_config = WirelessSensingConfig(
        frequency=args.frequency,
        bandwidth=args.bandwidth,
        num_subcarriers=args.num_subcarriers,
        tx_position=tx_position,
        rx_position=rx_position,
        frame_rate=30.0,  # Assuming 30 fps motion
    )

    # Initialize processors
    visualizer = SionnaRayVisualizer(ray_config)
    sensing_processor = WirelessSensingProcessor(sensing_config)

    print("\n[2/5] Generating motion and running ray tracing...")
    results = visualizer.generate_and_visualize(
        runtime=runtime,
        text_prompt=args.prompt,
        duration=args.duration,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        output_dir=args.output_dir,
    )

    print("\n[3/5] Processing wireless signals...")
    # Process each frame's ray paths into signal representations
    signals_list = []
    ray_paths = results.get("ray_paths", [])

    for frame_idx, paths_data in enumerate(ray_paths):
        signals = sensing_processor.process_paths(paths_data, frame_idx)
        signals_list.append(signals)

        # Extract features for this frame
        features = compute_signal_features(signals)

        if frame_idx == 0:
            print(f"  Frame {frame_idx}: RSSI={signals['rssi_dbm']:.1f} dBm, "
                  f"Paths={signals['num_paths']}, "
                  f"RMS Delay={signals['rms_delay_spread']*1e9:.2f} ns")

    print(f"  ... processed {len(signals_list)} frames total")

    print("\n[4/5] Analyzing signal representations...")
    # Compare different representations
    comparison = compare_signal_representations(signals_list)

    # Print analysis summary
    print_analysis_summary(comparison, signals_list)

    print("\n[5/5] Creating visualizations...")
    # Create signal analysis plot
    create_signal_analysis_plot(
        signals_list,
        os.path.join(args.output_dir, "signal_analysis.png"),
        args.prompt,
    )

    # Create representation comparison plot
    create_representation_comparison_plot(
        comparison,
        signals_list,
        os.path.join(args.output_dir, "representation_comparison.png"),
    )

    # Save dataset if requested
    if args.save_dataset:
        print("\nSaving ML-ready dataset...")
        dataset = MotionWirelessDataset(name="motion_sensing_single")
        dataset.add_sample(
            prompt=args.prompt,
            motion_data=results.get("motion_data", {}),
            wireless_signals=signals_list,
            config=sensing_config,
        )

        # Save as JSON
        dataset.save(os.path.join(args.output_dir, "dataset.json"))

        # Save as NumPy arrays
        dataset.save_numpy(os.path.join(args.output_dir, "numpy_data"))

    # Save analysis results
    analysis_results = {
        "prompt": args.prompt,
        "config": {
            "frequency": args.frequency,
            "bandwidth": args.bandwidth,
            "num_subcarriers": args.num_subcarriers,
            "tx_position": tx_position,
            "rx_position": rx_position,
            "duration": args.duration,
        },
        "temporal_variance": comparison.get("temporal_variance", {}),
        "recommendation": comparison.get("recommendation", ""),
        "num_frames": len(signals_list),
        "statistics": {
            "rssi_mean": float(np.mean([s["rssi_dbm"] for s in signals_list])),
            "rssi_std": float(np.std([s["rssi_dbm"] for s in signals_list])),
            "avg_num_paths": float(np.mean([s["num_paths"] for s in signals_list])),
        }
    }

    with open(os.path.join(args.output_dir, "analysis_results.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETED - Output files:")
    print("=" * 60)
    for f in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            print(f"  {f:40s} ({size/1024:.1f} KB)")
        else:
            print(f"  {f}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
