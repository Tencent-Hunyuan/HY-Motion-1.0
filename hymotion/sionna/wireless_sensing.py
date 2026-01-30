"""
Wireless Sensing Signal Processing for Motion-to-Wireless Simulation

This module provides various signal representations for wireless sensing:
- Channel Impulse Response (CIR)
- Channel Frequency Response (CFR/CSI)
- Doppler Profile
- Received Signal Strength (RSS)
- Phase Tracking
- MIMO Channel Matrix

These representations can be used to:
1. Analyze how human motion affects wireless channels
2. Train ML models for motion recognition from wireless signals
3. Generate synthetic wireless sensing datasets

Example:
    >>> from hymotion.sionna.wireless_sensing import WirelessSensingProcessor
    >>> processor = WirelessSensingProcessor(config)
    >>> signals = processor.process_frame(paths, frame_idx)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os


@dataclass
class WirelessSensingConfig:
    """Configuration for wireless sensing signal processing."""

    # RF parameters
    frequency: float = 3.5e9  # Carrier frequency (Hz)
    bandwidth: float = 100e6  # System bandwidth (Hz)
    num_subcarriers: int = 64  # Number of OFDM subcarriers

    # Time domain parameters
    cir_num_taps: int = 128  # Number of CIR taps
    cir_max_delay: float = 1e-6  # Maximum delay for CIR (seconds)

    # Antenna configuration
    num_tx_antennas: int = 4  # Number of TX antennas
    num_rx_antennas: int = 4  # Number of RX antennas

    # Doppler processing
    doppler_fft_size: int = 64  # FFT size for Doppler estimation
    frame_rate: float = 30.0  # Motion frame rate (Hz)

    # Noise
    add_noise: bool = False
    snr_db: float = 30.0  # Signal-to-noise ratio

    # TX/RX positions
    tx_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 3.0])
    rx_position: List[float] = field(default_factory=lambda: [5.0, 5.0, 1.5])


class WirelessSensingProcessor:
    """
    Process Sionna ray tracing results into various wireless sensing representations.

    This processor converts path-level data (delays, amplitudes, types) into
    standardized signal representations suitable for ML training.
    """

    def __init__(self, config: Optional[WirelessSensingConfig] = None):
        """Initialize the processor with configuration."""
        self.config = config or WirelessSensingConfig()

        # Pre-compute constants
        self.wavelength = 3e8 / self.config.frequency
        self.subcarrier_spacing = self.config.bandwidth / self.config.num_subcarriers
        self.tap_spacing = self.config.cir_max_delay / self.config.cir_num_taps

        # History for Doppler computation (across frames)
        self.channel_history: List[np.ndarray] = []
        self.max_history_len = self.config.doppler_fft_size

    def reset_history(self):
        """Reset channel history for new motion sequence."""
        self.channel_history = []

    def process_paths(
        self,
        paths_data: Dict[str, Any],
        frame_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Process ray tracing paths into wireless sensing signals.

        Args:
            paths_data: Dictionary containing:
                - vertices: Path coordinates
                - types: Path types (LoS, reflection, etc.)
                - delays: Propagation delays (seconds)
                - powers: Path powers (linear)
            frame_idx: Current frame index

        Returns:
            Dictionary containing all signal representations:
                - cir: Channel Impulse Response
                - cfr: Channel Frequency Response (CSI)
                - rssi: Received Signal Strength
                - phase: Phase per path
                - mimo_channel: Full MIMO channel matrix
                - doppler: Doppler spectrum (if enough history)
                - path_info: Raw path information
        """
        # Extract path data
        delays = paths_data.get("delays", np.array([]))
        powers = paths_data.get("powers", np.array([]))
        vertices = paths_data.get("vertices", [])
        path_types = paths_data.get("types", [])

        # Handle empty paths
        if len(delays) == 0 or (isinstance(delays, np.ndarray) and delays.size == 0):
            return self._create_empty_signals(frame_idx)

        # Flatten arrays if needed
        delays = np.atleast_1d(delays).flatten()
        powers = np.atleast_1d(powers).flatten()

        # Compute complex amplitudes (with random phase if not provided)
        amplitudes = np.sqrt(powers.astype(np.float64))
        phases = self._compute_phases_from_delays(delays)
        complex_amplitudes = amplitudes * np.exp(1j * phases)

        # Compute all signal representations
        results = {}

        # 1. Channel Impulse Response (CIR)
        results["cir"] = self._compute_cir(delays, complex_amplitudes)
        results["cir_magnitude"] = np.abs(results["cir"])
        results["cir_phase"] = np.angle(results["cir"])

        # 2. Channel Frequency Response (CFR/CSI)
        results["cfr"] = self._compute_cfr(results["cir"])
        results["cfr_magnitude"] = np.abs(results["cfr"])
        results["cfr_phase"] = np.angle(results["cfr"])

        # 3. RSSI (Received Signal Strength Indicator)
        results["rssi_linear"] = np.sum(powers)
        results["rssi_dbm"] = 10 * np.log10(results["rssi_linear"] + 1e-12) + 30  # Assuming 1mW ref

        # 4. Per-path information
        results["path_delays"] = delays
        results["path_powers"] = powers
        results["path_powers_db"] = 10 * np.log10(powers + 1e-12)
        results["path_phases"] = phases
        results["path_amplitudes"] = amplitudes
        results["num_paths"] = len(delays)

        # 5. MIMO Channel Matrix (simplified - assuming single link for now)
        results["mimo_channel"] = self._compute_mimo_channel(delays, complex_amplitudes)

        # 6. Path statistics
        results["mean_delay"] = np.average(delays, weights=powers) if len(delays) > 0 else 0
        results["rms_delay_spread"] = self._compute_rms_delay_spread(delays, powers)
        results["coherence_bandwidth"] = 1 / (2 * np.pi * results["rms_delay_spread"]) if results["rms_delay_spread"] > 0 else np.inf

        # 7. Update history and compute Doppler
        self.channel_history.append(results["cfr"].copy())
        if len(self.channel_history) > self.max_history_len:
            self.channel_history.pop(0)

        if len(self.channel_history) >= 2:
            results["doppler_spectrum"] = self._compute_doppler_spectrum()
            results["doppler_shift"] = self._estimate_doppler_shift()
        else:
            results["doppler_spectrum"] = np.zeros(self.config.doppler_fft_size)
            results["doppler_shift"] = 0.0

        # 8. Frame metadata
        results["frame_idx"] = frame_idx
        results["timestamp"] = frame_idx / self.config.frame_rate

        # Add noise if configured
        if self.config.add_noise:
            results = self._add_noise(results)

        return results

    def _compute_cir(
        self,
        delays: np.ndarray,
        complex_amplitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Channel Impulse Response.

        CIR h(τ) = Σ a_i * δ(τ - τ_i)

        Discretized into taps based on delay bins.
        """
        cir = np.zeros(self.config.cir_num_taps, dtype=np.complex128)

        for delay, amplitude in zip(delays, complex_amplitudes):
            # Map delay to tap index
            tap_idx = int(delay / self.tap_spacing)
            if 0 <= tap_idx < self.config.cir_num_taps:
                cir[tap_idx] += amplitude

        return cir

    def _compute_cfr(self, cir: np.ndarray) -> np.ndarray:
        """
        Compute Channel Frequency Response (CSI) via FFT.

        CFR H(f) = FFT(h(τ))
        """
        # Zero-pad CIR to match number of subcarriers
        padded_cir = np.zeros(self.config.num_subcarriers, dtype=np.complex128)
        padded_cir[:min(len(cir), self.config.num_subcarriers)] = cir[:self.config.num_subcarriers]

        # FFT to get frequency response
        cfr = np.fft.fft(padded_cir)

        return cfr

    def _compute_mimo_channel(
        self,
        delays: np.ndarray,
        complex_amplitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MIMO channel matrix H.

        For MIMO, each TX-RX pair has a different phase shift based on
        antenna spacing and angle of arrival/departure.

        Returns:
            H: (num_rx, num_tx, num_subcarriers) complex channel matrix
        """
        num_rx = self.config.num_rx_antennas
        num_tx = self.config.num_tx_antennas
        num_sc = self.config.num_subcarriers

        H = np.zeros((num_rx, num_tx, num_sc), dtype=np.complex128)

        # Simplified model: each path contributes to all antenna pairs
        # with additional phase shifts based on array geometry
        d_ant = self.wavelength / 2  # Half-wavelength spacing

        for path_idx, (delay, amp) in enumerate(zip(delays, complex_amplitudes)):
            # Random angle of arrival/departure for each path (simplified)
            aoa = np.random.uniform(-np.pi/2, np.pi/2)
            aod = np.random.uniform(-np.pi/2, np.pi/2)

            # Array response vectors
            a_rx = np.exp(1j * 2 * np.pi * d_ant / self.wavelength *
                         np.arange(num_rx) * np.sin(aoa))
            a_tx = np.exp(1j * 2 * np.pi * d_ant / self.wavelength *
                         np.arange(num_tx) * np.sin(aod))

            # Frequency response for this path
            freq_response = np.exp(-1j * 2 * np.pi *
                                   np.arange(num_sc) * self.subcarrier_spacing * delay)

            # Outer product to form contribution to H
            for sc in range(num_sc):
                H[:, :, sc] += amp * freq_response[sc] * np.outer(a_rx, a_tx.conj())

        return H

    def _compute_doppler_spectrum(self) -> np.ndarray:
        """
        Compute Doppler spectrum from channel history.

        Doppler is computed by taking FFT across time dimension of CFR.
        """
        if len(self.channel_history) < 2:
            return np.zeros(self.config.doppler_fft_size)

        # Stack channel history into time-frequency matrix
        cfr_history = np.array(self.channel_history)  # (time, subcarriers)

        # Zero-pad to FFT size
        if len(cfr_history) < self.config.doppler_fft_size:
            pad_len = self.config.doppler_fft_size - len(cfr_history)
            cfr_history = np.pad(cfr_history, ((0, pad_len), (0, 0)), mode='constant')

        # FFT across time for each subcarrier
        doppler_per_sc = np.fft.fft(cfr_history, axis=0)

        # Average magnitude across subcarriers
        doppler_spectrum = np.mean(np.abs(doppler_per_sc), axis=1)

        # Shift zero-Doppler to center
        doppler_spectrum = np.fft.fftshift(doppler_spectrum)

        return doppler_spectrum

    def _estimate_doppler_shift(self) -> float:
        """
        Estimate dominant Doppler shift from channel phase changes.
        """
        if len(self.channel_history) < 2:
            return 0.0

        # Phase difference between consecutive frames
        phase_diff = np.angle(
            self.channel_history[-1] * np.conj(self.channel_history[-2])
        )

        # Average phase change
        avg_phase_diff = np.mean(phase_diff)

        # Convert to Doppler frequency
        # Δφ = 2π * f_d * Δt
        dt = 1.0 / self.config.frame_rate
        doppler_shift = avg_phase_diff / (2 * np.pi * dt)

        return float(doppler_shift)

    def _compute_phases_from_delays(self, delays: np.ndarray) -> np.ndarray:
        """Compute phase shifts from propagation delays."""
        # φ = 2π * f * τ
        phases = 2 * np.pi * self.config.frequency * delays
        # Wrap to [-π, π]
        phases = np.mod(phases + np.pi, 2 * np.pi) - np.pi
        return phases

    def _compute_rms_delay_spread(
        self,
        delays: np.ndarray,
        powers: np.ndarray,
    ) -> float:
        """Compute RMS delay spread."""
        if len(delays) == 0 or np.sum(powers) == 0:
            return 0.0

        # Normalize powers
        powers_norm = powers / np.sum(powers)

        # Mean delay
        mean_delay = np.sum(delays * powers_norm)

        # RMS delay spread
        rms_spread = np.sqrt(np.sum(powers_norm * (delays - mean_delay)**2))

        return float(rms_spread)

    def _add_noise(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add AWGN noise to signals."""
        snr_linear = 10 ** (self.config.snr_db / 10)

        # Add noise to CIR
        cir = results["cir"]
        signal_power = np.mean(np.abs(cir)**2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(cir)) + 1j * np.random.randn(len(cir))
        )
        results["cir_noisy"] = cir + noise

        # Add noise to CFR
        cfr = results["cfr"]
        signal_power = np.mean(np.abs(cfr)**2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(cfr)) + 1j * np.random.randn(len(cfr))
        )
        results["cfr_noisy"] = cfr + noise

        return results

    def _create_empty_signals(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Create empty signal structure when no paths are found."""
        return {
            "cir": np.zeros(self.config.cir_num_taps, dtype=np.complex128),
            "cir_magnitude": np.zeros(self.config.cir_num_taps),
            "cir_phase": np.zeros(self.config.cir_num_taps),
            "cfr": np.zeros(self.config.num_subcarriers, dtype=np.complex128),
            "cfr_magnitude": np.zeros(self.config.num_subcarriers),
            "cfr_phase": np.zeros(self.config.num_subcarriers),
            "rssi_linear": 0.0,
            "rssi_dbm": -120.0,  # Very low
            "path_delays": np.array([]),
            "path_powers": np.array([]),
            "path_powers_db": np.array([]),
            "path_phases": np.array([]),
            "path_amplitudes": np.array([]),
            "num_paths": 0,
            "mimo_channel": np.zeros(
                (self.config.num_rx_antennas, self.config.num_tx_antennas,
                 self.config.num_subcarriers), dtype=np.complex128
            ),
            "mean_delay": 0.0,
            "rms_delay_spread": 0.0,
            "coherence_bandwidth": np.inf,
            "doppler_spectrum": np.zeros(self.config.doppler_fft_size),
            "doppler_shift": 0.0,
            "frame_idx": frame_idx,
            "timestamp": frame_idx / self.config.frame_rate,
        }


class MotionWirelessDataset:
    """
    Dataset class for motion-to-wireless mapping.

    Stores motion data and corresponding wireless signals for ML training.
    """

    def __init__(self, name: str = "motion_wireless"):
        """Initialize dataset."""
        self.name = name
        self.samples: List[Dict[str, Any]] = []
        self.metadata = {
            "name": name,
            "version": "1.0",
            "description": "Motion to wireless sensing dataset",
            "signal_representations": [
                "cir", "cfr", "doppler", "rssi", "phase", "mimo_channel"
            ],
        }

    def add_sample(
        self,
        prompt: str,
        motion_data: Dict[str, Any],
        wireless_signals: List[Dict[str, np.ndarray]],
        config: WirelessSensingConfig,
    ):
        """
        Add a motion-wireless pair to the dataset.

        Args:
            prompt: Text prompt that generated the motion
            motion_data: Motion generation output (rot6d, transl, keypoints3d)
            wireless_signals: List of signal dicts per frame
            config: Wireless sensing configuration used
        """
        sample = {
            "prompt": prompt,
            "motion": self._serialize_motion(motion_data),
            "signals": self._serialize_signals(wireless_signals),
            "config": {
                "frequency": config.frequency,
                "bandwidth": config.bandwidth,
                "num_subcarriers": config.num_subcarriers,
                "tx_position": config.tx_position,
                "rx_position": config.rx_position,
                "frame_rate": config.frame_rate,
            },
            "num_frames": len(wireless_signals),
        }
        self.samples.append(sample)

    def _serialize_motion(self, motion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert motion data to serializable format."""
        result = {}
        for key, value in motion_data.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif hasattr(value, 'cpu'):  # torch tensor
                result[key] = value.cpu().numpy().tolist()
            else:
                result[key] = value
        return result

    def _serialize_signals(
        self,
        signals: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Convert signal data to serializable format."""
        result = []
        for frame_signals in signals:
            frame_data = {}
            for key, value in frame_signals.items():
                if isinstance(value, np.ndarray):
                    if np.iscomplexobj(value):
                        # Store complex as magnitude and phase
                        frame_data[f"{key}_mag"] = np.abs(value).tolist()
                        frame_data[f"{key}_phase"] = np.angle(value).tolist()
                    else:
                        frame_data[key] = value.tolist()
                elif isinstance(value, (int, float, str)):
                    frame_data[key] = value
            result.append(frame_data)
        return result

    def save(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            "metadata": self.metadata,
            "samples": self.samples,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Dataset saved to {filepath}")

    def save_numpy(self, output_dir: str):
        """
        Save dataset as NumPy arrays for efficient ML training.

        Creates separate .npy files for different signal representations.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Collect all data
        prompts = []
        cir_data = []
        cfr_data = []
        doppler_data = []
        rssi_data = []

        for sample in self.samples:
            prompts.append(sample["prompt"])

            # Stack signals across frames
            frame_cirs = []
            frame_cfrs = []
            frame_dopplers = []
            frame_rssis = []

            for frame in sample["signals"]:
                if "cir_mag" in frame:
                    cir_complex = np.array(frame["cir_mag"]) * np.exp(
                        1j * np.array(frame.get("cir_phase", np.zeros_like(frame["cir_mag"])))
                    )
                    frame_cirs.append(cir_complex)

                if "cfr_mag" in frame:
                    cfr_complex = np.array(frame["cfr_mag"]) * np.exp(
                        1j * np.array(frame.get("cfr_phase", np.zeros_like(frame["cfr_mag"])))
                    )
                    frame_cfrs.append(cfr_complex)

                if "doppler_spectrum" in frame:
                    frame_dopplers.append(np.array(frame["doppler_spectrum"]))

                if "rssi_dbm" in frame:
                    frame_rssis.append(frame["rssi_dbm"])

            if frame_cirs:
                cir_data.append(np.array(frame_cirs))
            if frame_cfrs:
                cfr_data.append(np.array(frame_cfrs))
            if frame_dopplers:
                doppler_data.append(np.array(frame_dopplers))
            if frame_rssis:
                rssi_data.append(np.array(frame_rssis))

        # Save arrays
        if cir_data:
            np.save(os.path.join(output_dir, "cir.npy"), np.array(cir_data, dtype=object))
        if cfr_data:
            np.save(os.path.join(output_dir, "cfr.npy"), np.array(cfr_data, dtype=object))
        if doppler_data:
            np.save(os.path.join(output_dir, "doppler.npy"), np.array(doppler_data, dtype=object))
        if rssi_data:
            np.save(os.path.join(output_dir, "rssi.npy"), np.array(rssi_data, dtype=object))

        # Save prompts
        with open(os.path.join(output_dir, "prompts.json"), 'w') as f:
            json.dump(prompts, f, indent=2)

        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"NumPy dataset saved to {output_dir}")

    @classmethod
    def load(cls, filepath: str) -> 'MotionWirelessDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        dataset = cls(data["metadata"]["name"])
        dataset.metadata = data["metadata"]
        dataset.samples = data["samples"]
        return dataset


def compute_signal_features(signals: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Extract ML-friendly features from wireless signals.

    These features can be used as input to classification/regression models.

    Returns:
        Dictionary of scalar features
    """
    features = {}

    # CIR features
    if "cir_magnitude" in signals:
        cir_mag = signals["cir_magnitude"]
        features["cir_peak"] = float(np.max(cir_mag))
        features["cir_mean"] = float(np.mean(cir_mag))
        features["cir_std"] = float(np.std(cir_mag))
        features["cir_energy"] = float(np.sum(cir_mag**2))

        # Peak position (related to LoS delay)
        features["cir_peak_idx"] = float(np.argmax(cir_mag))

        # Number of significant taps
        threshold = 0.1 * np.max(cir_mag)
        features["cir_num_significant_taps"] = float(np.sum(cir_mag > threshold))

    # CFR features
    if "cfr_magnitude" in signals:
        cfr_mag = signals["cfr_magnitude"]
        features["cfr_mean"] = float(np.mean(cfr_mag))
        features["cfr_std"] = float(np.std(cfr_mag))
        features["cfr_flatness"] = float(np.exp(np.mean(np.log(cfr_mag + 1e-12))) / (np.mean(cfr_mag) + 1e-12))

    # Doppler features
    if "doppler_spectrum" in signals:
        doppler = signals["doppler_spectrum"]
        features["doppler_peak"] = float(np.max(doppler))
        features["doppler_mean"] = float(np.mean(doppler))

        # Doppler spread
        doppler_norm = doppler / (np.sum(doppler) + 1e-12)
        freq_bins = np.arange(len(doppler)) - len(doppler) // 2
        features["doppler_mean_shift"] = float(np.sum(freq_bins * doppler_norm))
        features["doppler_spread"] = float(np.sqrt(np.sum(doppler_norm * (freq_bins - features["doppler_mean_shift"])**2)))

    # RSSI
    if "rssi_dbm" in signals:
        features["rssi"] = float(signals["rssi_dbm"])

    # Delay spread
    if "rms_delay_spread" in signals:
        features["rms_delay_spread"] = float(signals["rms_delay_spread"])
        features["coherence_bandwidth"] = float(signals.get("coherence_bandwidth", 0))

    # Path count
    if "num_paths" in signals:
        features["num_paths"] = float(signals["num_paths"])

    return features


def compare_signal_representations(
    signals_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Compare different signal representations across frames.

    Useful for determining which representation best captures motion.

    Args:
        signals_list: List of signal dicts from multiple frames

    Returns:
        Dictionary containing:
        - temporal_variance: How much each representation varies over time
        - correlation_matrix: Correlations between representations
    """
    num_frames = len(signals_list)

    if num_frames < 2:
        return {"temporal_variance": {}, "recommendation": "Need more frames"}

    # Collect time series for each representation
    representations = {
        "rssi": [],
        "cir_energy": [],
        "cfr_mean": [],
        "doppler_peak": [],
        "phase_std": [],
    }

    for signals in signals_list:
        if "rssi_dbm" in signals:
            representations["rssi"].append(signals["rssi_dbm"])
        if "cir_magnitude" in signals:
            representations["cir_energy"].append(np.sum(signals["cir_magnitude"]**2))
        if "cfr_magnitude" in signals:
            representations["cfr_mean"].append(np.mean(signals["cfr_magnitude"]))
        if "doppler_spectrum" in signals:
            representations["doppler_peak"].append(np.max(signals["doppler_spectrum"]))
        if "cfr_phase" in signals:
            representations["phase_std"].append(np.std(signals["cfr_phase"]))

    # Compute temporal variance (higher = more sensitive to motion)
    temporal_variance = {}
    for name, values in representations.items():
        if len(values) > 1:
            values = np.array(values)
            # Normalize by mean to get coefficient of variation
            mean_val = np.mean(np.abs(values))
            if mean_val > 0:
                temporal_variance[name] = float(np.std(values) / mean_val)
            else:
                temporal_variance[name] = 0.0

    # Rank representations by sensitivity
    ranked = sorted(temporal_variance.items(), key=lambda x: x[1], reverse=True)

    recommendation = (
        f"Most motion-sensitive: {ranked[0][0]} (CV={ranked[0][1]:.3f})\n"
        if ranked else "No data"
    )
    for name, cv in ranked[1:]:
        recommendation += f"  {name}: CV={cv:.3f}\n"

    return {
        "temporal_variance": temporal_variance,
        "ranked_representations": ranked,
        "recommendation": recommendation,
    }
