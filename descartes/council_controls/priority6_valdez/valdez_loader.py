"""
valdez_loader.py

Phase 6-1: Load Valdez et al. 2022 Dataset from OSF nf7s8

Dataset: Human intracranial EEG during active emotion task.
Patients performed emotion recognition/discrimination tasks while
single-unit and LFP recordings were made from amygdala, hippocampus,
and prefrontal regions.

Key data structure:
- Trial-level spike counts in TWO windows per trial:
    1. Stimulus window (image onset to response)
    2. Response window (response to end of trial)
- LFP features (theta, gamma, broadband) per trial per electrode
- Trial metadata (emotion category, response, RT, patient ID)

Converts to unified HDF5 format compatible with DESCARTES pipeline:
    /patient_XX/
        input_spikes    (n_trials, n_input_neurons, 2)  # 2 windows
        output_spikes   (n_trials, n_output_neurons, 2)
        lfp_features    (n_trials, n_channels, n_features)
        trial_metadata  (n_trials,) compound dtype
        electrode_info  dataset with region labels

No external dependencies beyond stdlib + json + dataclasses.
Actual I/O requires h5py and numpy at runtime.

Usage:
    python -m descartes.council_controls.priority6_valdez.valdez_loader
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..config_council import QAP


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElectrodeInfo:
    electrode_id: str
    patient_id: str
    brain_region: str
    hemisphere: str
    is_input: bool  # True = source region, False = target region
    channel_index: int


@dataclass
class TrialMetadata:
    trial_id: int
    patient_id: str
    emotion_category: str
    task_type: str       # "recognition" or "discrimination"
    response: str        # patient response
    reaction_time_ms: float
    correct: bool
    stimulus_window_ms: Tuple[float, float] = (0.0, 0.0)
    response_window_ms: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ValdezPatientData:
    patient_id: str
    n_trials: int
    n_input_neurons: int
    n_output_neurons: int
    electrodes: List[ElectrodeInfo] = field(default_factory=list)
    trials: List[TrialMetadata] = field(default_factory=list)
    # These hold numpy arrays at runtime; typed as Optional for stdlib compat
    input_spikes: Optional[object] = None   # (n_trials, n_input, 2)
    output_spikes: Optional[object] = None  # (n_trials, n_output, 2)
    lfp_features: Optional[object] = None   # (n_trials, n_channels, n_feat)


@dataclass
class ValdezDataset:
    osf_project: str
    patients: List[ValdezPatientData] = field(default_factory=list)
    n_patients: int = 0
    task_type: str = "active_emotion"
    source_regions: List[str] = field(default_factory=lambda: [
        "amygdala", "hippocampus"
    ])
    target_regions: List[str] = field(default_factory=lambda: [
        "prefrontal", "orbitofrontal", "anterior_cingulate"
    ])


# ---------------------------------------------------------------------------
# LFP feature extraction (per trial, per channel)
# ---------------------------------------------------------------------------

LFP_FEATURE_NAMES = [
    "theta_power",       # 4-8 Hz band power
    "gamma_amp",         # 30-80 Hz amplitude
    "broadband_power",   # 1-200 Hz total power
    "theta_gamma_pac",   # phase-amplitude coupling
    "firing_rate",       # spike count / window duration
    "synchrony",         # pairwise phase consistency
]


def compute_trial_lfp_features(lfp_segment, fs: float) -> list:
    """Compute LFP features for a single trial segment.

    Parameters
    ----------
    lfp_segment : array-like, shape (n_samples, n_channels)
        Raw LFP data for one trial.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    features : list of float
        One value per feature in LFP_FEATURE_NAMES.

    NOTE: Actual spectral computation requires numpy/scipy at runtime.
    This stub returns zeros for stdlib-only execution.
    """
    try:
        import numpy as np
        n_samples, n_channels = lfp_segment.shape
        duration_s = n_samples / fs

        features = []
        # theta_power: bandpass 4-8 Hz, sum of squared amplitudes
        features.append(0.0)  # placeholder — real impl uses scipy.signal
        # gamma_amp: bandpass 30-80 Hz, mean amplitude
        features.append(0.0)
        # broadband_power
        features.append(float(np.mean(lfp_segment ** 2)))
        # theta_gamma_pac
        features.append(0.0)
        # firing_rate (computed from spike data, not LFP)
        features.append(0.0)
        # synchrony
        features.append(0.0)

        return features
    except ImportError:
        return [0.0] * len(LFP_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Raw data parsing
# ---------------------------------------------------------------------------

def parse_spike_counts_two_windows(spike_data: dict) -> Tuple[list, list]:
    """Parse trial-level spike counts into stimulus and response windows.

    Valdez data provides spike counts in two temporal windows per trial:
        1. Stimulus window: image onset -> button press
        2. Response window: button press -> trial end

    Parameters
    ----------
    spike_data : dict
        Raw spike data dictionary from the dataset.
        Expected keys: 'stimulus_counts', 'response_counts'
        Each is a list of spike counts per neuron.

    Returns
    -------
    stim_counts, resp_counts : list, list
        Spike counts for each window.
    """
    stim_counts = spike_data.get('stimulus_counts', [])
    resp_counts = spike_data.get('response_counts', [])
    return stim_counts, resp_counts


def classify_electrode_region(region_label: str,
                              source_regions: list,
                              target_regions: list) -> bool:
    """Determine if electrode is in input (source) or output (target) region.

    Returns True if input (source), False if output (target).
    Raises ValueError if region not recognised.
    """
    label_lower = region_label.lower()
    for sr in source_regions:
        if sr.lower() in label_lower:
            return True
    for tr in target_regions:
        if tr.lower() in label_lower:
            return False
    raise ValueError(
        "Electrode region '{}' not in source {} or target {}".format(
            region_label, source_regions, target_regions))


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_valdez_from_osf(data_dir: str,
                         min_trials: int = None) -> ValdezDataset:
    """Load Valdez et al. 2022 data from local copy of OSF nf7s8.

    Parameters
    ----------
    data_dir : str
        Path to local directory containing the downloaded OSF data.
        Expected structure:
            data_dir/
                patient_XX/
                    spikes.json     (trial-level spike counts)
                    lfp.h5          (raw LFP per trial)
                    metadata.json   (trial info, electrode info)
    min_trials : int, optional
        Minimum trials per patient. Defaults to QAP.valdez_min_trials_per_patient.

    Returns
    -------
    ValdezDataset
    """
    if min_trials is None:
        min_trials = QAP.valdez_min_trials_per_patient

    dataset = ValdezDataset(osf_project=QAP.valdez_osf_project)
    data_path = Path(data_dir)

    if not data_path.exists():
        print("WARNING: Data directory '{}' not found. "
              "Returning empty dataset.".format(data_dir))
        return dataset

    patient_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("patient")
    ])

    for pdir in patient_dirs:
        patient_id = pdir.name
        metadata_file = pdir / "metadata.json"
        spikes_file = pdir / "spikes.json"

        if not metadata_file.exists():
            print("  SKIP {}: no metadata.json".format(patient_id))
            continue

        with open(metadata_file) as f:
            meta = json.load(f)

        # Parse electrode info
        electrodes = []
        source_regions = dataset.source_regions
        target_regions = dataset.target_regions
        for i, einfo in enumerate(meta.get('electrodes', [])):
            region = einfo.get('region', 'unknown')
            try:
                is_input = classify_electrode_region(
                    region, source_regions, target_regions)
            except ValueError:
                continue
            electrodes.append(ElectrodeInfo(
                electrode_id=einfo.get('id', 'e{}'.format(i)),
                patient_id=patient_id,
                brain_region=region,
                hemisphere=einfo.get('hemisphere', 'unknown'),
                is_input=is_input,
                channel_index=i,
            ))

        # Parse trial metadata
        trials = []
        for t in meta.get('trials', []):
            trials.append(TrialMetadata(
                trial_id=t.get('trial_id', 0),
                patient_id=patient_id,
                emotion_category=t.get('emotion', 'unknown'),
                task_type=t.get('task_type', 'recognition'),
                response=t.get('response', ''),
                reaction_time_ms=t.get('rt_ms', 0.0),
                correct=t.get('correct', False),
                stimulus_window_ms=tuple(t.get('stim_window', [0, 0])),
                response_window_ms=tuple(t.get('resp_window', [0, 0])),
            ))

        if len(trials) < min_trials:
            print("  SKIP {}: only {} trials (need {})".format(
                patient_id, len(trials), min_trials))
            continue

        n_input = sum(1 for e in electrodes if e.is_input)
        n_output = sum(1 for e in electrodes if not e.is_input)

        patient_data = ValdezPatientData(
            patient_id=patient_id,
            n_trials=len(trials),
            n_input_neurons=n_input,
            n_output_neurons=n_output,
            electrodes=electrodes,
            trials=trials,
        )

        # Load spike counts if available
        if spikes_file.exists():
            patient_data = _load_spike_arrays(patient_data, spikes_file)

        dataset.patients.append(patient_data)

    dataset.n_patients = len(dataset.patients)
    return dataset


def _load_spike_arrays(patient_data: ValdezPatientData,
                       spikes_file: Path) -> ValdezPatientData:
    """Load spike count arrays from JSON and attach to patient data."""
    try:
        import numpy as np
    except ImportError:
        return patient_data

    with open(spikes_file) as f:
        spike_json = json.load(f)

    n_trials = patient_data.n_trials
    n_input = patient_data.n_input_neurons
    n_output = patient_data.n_output_neurons

    input_spikes = np.zeros((n_trials, n_input, 2), dtype=np.float32)
    output_spikes = np.zeros((n_trials, n_output, 2), dtype=np.float32)

    for t_idx, trial_spikes in enumerate(spike_json.get('trials', [])):
        if t_idx >= n_trials:
            break
        stim, resp = parse_spike_counts_two_windows(trial_spikes)
        # Separate input vs output neurons
        inp_idx = 0
        out_idx = 0
        for n_idx in range(len(stim)):
            if n_idx < n_input:
                input_spikes[t_idx, inp_idx, 0] = stim[n_idx] if n_idx < len(stim) else 0
                input_spikes[t_idx, inp_idx, 1] = resp[n_idx] if n_idx < len(resp) else 0
                inp_idx += 1
            else:
                oi = n_idx - n_input
                if oi < n_output:
                    output_spikes[t_idx, oi, 0] = stim[n_idx] if n_idx < len(stim) else 0
                    output_spikes[t_idx, oi, 1] = resp[n_idx] if n_idx < len(resp) else 0
                    out_idx += 1

    patient_data.input_spikes = input_spikes
    patient_data.output_spikes = output_spikes
    return patient_data


# ---------------------------------------------------------------------------
# Conversion to unified HDF5
# ---------------------------------------------------------------------------

def convert_to_unified_hdf5(dataset: ValdezDataset,
                            output_path: str) -> str:
    """Convert ValdezDataset to unified HDF5 format.

    Output structure:
        /patient_XX/
            input_spikes    (n_trials, n_input, 2)
            output_spikes   (n_trials, n_output, 2)
            lfp_features    (n_trials, n_channels, n_features)
            trial_emotion   (n_trials,) string
            trial_correct   (n_trials,) bool
            trial_rt_ms     (n_trials,) float
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("WARNING: h5py/numpy not available. Saving metadata as JSON.")
        return _save_metadata_json(dataset, output_path)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), 'w') as hf:
        hf.attrs['dataset'] = 'Valdez et al. 2022'
        hf.attrs['osf_project'] = dataset.osf_project
        hf.attrs['task_type'] = dataset.task_type
        hf.attrs['n_patients'] = dataset.n_patients
        hf.attrs['source_regions'] = json.dumps(dataset.source_regions)
        hf.attrs['target_regions'] = json.dumps(dataset.target_regions)

        for patient in dataset.patients:
            grp = hf.create_group(patient.patient_id)
            grp.attrs['n_trials'] = patient.n_trials
            grp.attrs['n_input_neurons'] = patient.n_input_neurons
            grp.attrs['n_output_neurons'] = patient.n_output_neurons

            if patient.input_spikes is not None:
                grp.create_dataset('input_spikes', data=patient.input_spikes)
            if patient.output_spikes is not None:
                grp.create_dataset('output_spikes', data=patient.output_spikes)
            if patient.lfp_features is not None:
                grp.create_dataset('lfp_features', data=patient.lfp_features)

            # Trial-level metadata
            emotions = [t.emotion_category for t in patient.trials]
            corrects = [t.correct for t in patient.trials]
            rts = [t.reaction_time_ms for t in patient.trials]
            grp.create_dataset('trial_emotion',
                               data=[e.encode('utf-8') for e in emotions])
            grp.create_dataset('trial_correct', data=corrects)
            grp.create_dataset('trial_rt_ms', data=rts)

            # Electrode metadata as JSON attribute
            elec_info = [asdict(e) for e in patient.electrodes]
            grp.attrs['electrode_info'] = json.dumps(elec_info)

    print("Unified HDF5 saved: {}".format(out_path))
    return str(out_path)


def _save_metadata_json(dataset: ValdezDataset, output_path: str) -> str:
    """Fallback: save metadata as JSON when h5py unavailable."""
    out_path = Path(output_path).with_suffix('.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        'dataset': 'Valdez et al. 2022',
        'osf_project': dataset.osf_project,
        'task_type': dataset.task_type,
        'n_patients': dataset.n_patients,
        'patients': [],
    }
    for p in dataset.patients:
        pmeta = {
            'patient_id': p.patient_id,
            'n_trials': p.n_trials,
            'n_input_neurons': p.n_input_neurons,
            'n_output_neurons': p.n_output_neurons,
            'electrodes': [asdict(e) for e in p.electrodes],
            'trials': [asdict(t) for t in p.trials],
        }
        meta['patients'].append(pmeta)

    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("Metadata JSON saved: {}".format(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_dataset_summary(dataset: ValdezDataset):
    """Print summary table of loaded Valdez dataset."""
    print("\n" + "=" * 70)
    print("VALDEZ et al. 2022 — ACTIVE EMOTION DATASET")
    print("OSF project: {}".format(dataset.osf_project))
    print("=" * 70)
    print("{:<15} | {:>8} | {:>8} | {:>8} | {}".format(
        "Patient", "Trials", "Input N", "Output N", "Task"))
    print("-" * 70)
    for p in dataset.patients:
        task_types = set(t.task_type for t in p.trials)
        print("{:<15} | {:>8} | {:>8} | {:>8} | {}".format(
            p.patient_id, p.n_trials, p.n_input_neurons,
            p.n_output_neurons, ", ".join(task_types)))
    print("-" * 70)
    print("Total patients: {}".format(dataset.n_patients))
    total_trials = sum(p.n_trials for p in dataset.patients)
    print("Total trials:   {}".format(total_trials))
    print("=" * 70)


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/valdez_osf_nf7s8"
    ds = load_valdez_from_osf(data_dir)
    print_dataset_summary(ds)
