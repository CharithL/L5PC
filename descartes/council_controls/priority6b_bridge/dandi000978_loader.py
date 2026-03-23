"""
dandi000978_loader.py

Phase 6B-1: Load DANDI 000978 — Rat CA1 + PFC W-Maze Data

Dataset: Hippocampal CA1 and prefrontal cortex (PFC) simultaneous
recordings from rats performing a W-maze spatial alternation task.

Key properties:
    - Input region: CA1 (hippocampal place cells)
    - Output region: PFC (prefrontal cortex)
    - Task: W-maze spatial alternation (memory-guided)
    - Species: rat (no human consent issues)
    - Format: NWB (Neurodata Without Borders)

Converts NWB to unified HDF5 format compatible with DESCARTES pipeline:
    /session_XX/
        ca1_spikes      (n_timesteps, n_ca1_neurons)
        pfc_spikes      (n_timesteps, n_pfc_neurons)
        lfp_ca1         (n_timesteps_lfp, n_ca1_channels)
        lfp_pfc         (n_timesteps_lfp, n_pfc_channels)
        position        (n_timesteps_pos, 2)  x,y
        trial_info      compound dtype
        time_bins       (n_timesteps,)

No external dependencies beyond stdlib + json + dataclasses.
Actual I/O requires h5py, numpy, pynwb at runtime.

Usage:
    python -m descartes.council_controls.priority6b_bridge.dandi000978_loader \\
        --data_dir data/dandi_000978
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
class UnitInfo:
    unit_id: str
    brain_region: str  # "CA1" or "PFC"
    electrode_id: str
    quality: str       # "good", "mua", "noise"
    mean_firing_rate: float
    is_input: bool     # True for CA1 (input), False for PFC (output)


@dataclass
class SessionInfo:
    session_id: str
    animal_id: str
    date: str
    n_ca1_units: int
    n_pfc_units: int
    n_trials: int
    duration_s: float
    sampling_rate_spikes: float
    sampling_rate_lfp: float
    maze_type: str = "W-maze"


@dataclass
class SessionData:
    """Loaded data for one recording session."""
    session_info: SessionInfo
    units: List[UnitInfo] = field(default_factory=list)
    # Arrays populated at runtime (numpy)
    ca1_spikes: Optional[object] = None     # (n_timesteps, n_ca1)
    pfc_spikes: Optional[object] = None     # (n_timesteps, n_pfc)
    lfp_ca1: Optional[object] = None        # (n_lfp_samples, n_ca1_ch)
    lfp_pfc: Optional[object] = None        # (n_lfp_samples, n_pfc_ch)
    position: Optional[object] = None       # (n_pos_samples, 2)
    time_bins: Optional[object] = None      # (n_timesteps,)
    trial_starts: Optional[object] = None   # (n_trials,)
    trial_ends: Optional[object] = None     # (n_trials,)


@dataclass
class DANDI000978Dataset:
    """Full dataset from DANDI 000978."""
    dandi_id: str = "000978"
    input_region: str = "CA1"
    output_region: str = "PFC"
    species: str = "rat"
    task: str = "W-maze spatial alternation"
    sessions: List[SessionData] = field(default_factory=list)
    n_sessions: int = 0


# ---------------------------------------------------------------------------
# NWB parsing helpers
# ---------------------------------------------------------------------------

def classify_unit_region(unit_meta: dict) -> Tuple[str, bool]:
    """Classify a unit as CA1 (input) or PFC (output).

    Parameters
    ----------
    unit_meta : dict
        Unit metadata from NWB file. Expected keys include
        'brain_area', 'location', or 'electrode_group'.

    Returns
    -------
    region : str
        'CA1' or 'PFC'
    is_input : bool
        True for CA1, False for PFC
    """
    # Check various metadata fields for region info
    for key in ('brain_area', 'location', 'electrode_group', 'region'):
        val = str(unit_meta.get(key, '')).upper()
        if 'CA1' in val or 'HIPPOCAMPUS' in val or 'HPC' in val:
            return 'CA1', True
        if 'PFC' in val or 'PREFRONTAL' in val or 'MPFC' in val or 'DLPFC' in val:
            return 'PFC', False

    # Default: unknown — skip this unit
    raise ValueError("Cannot classify unit region from metadata: {}".format(
        unit_meta))


def bin_spikes(spike_times: list, time_bins: list) -> list:
    """Bin spike times into time bins.

    Parameters
    ----------
    spike_times : list of float
        Timestamps of individual spikes.
    time_bins : list of float
        Bin edges (n_bins + 1 values).

    Returns
    -------
    counts : list of int
        Spike count in each bin.
    """
    n_bins = len(time_bins) - 1
    counts = [0] * n_bins
    spike_idx = 0
    n_spikes = len(spike_times)

    for b in range(n_bins):
        t_start = time_bins[b]
        t_end = time_bins[b + 1]
        while spike_idx < n_spikes and spike_times[spike_idx] < t_start:
            spike_idx += 1
        while spike_idx < n_spikes and spike_times[spike_idx] < t_end:
            counts[b] += 1
            spike_idx += 1

    return counts


# ---------------------------------------------------------------------------
# NWB loader
# ---------------------------------------------------------------------------

def load_nwb_session(nwb_path: str,
                     bin_size_s: float = 0.025) -> Optional[SessionData]:
    """Load a single NWB file and extract CA1 and PFC data.

    Parameters
    ----------
    nwb_path : str
        Path to the NWB file.
    bin_size_s : float
        Spike binning resolution in seconds (default 25ms).

    Returns
    -------
    SessionData or None
    """
    try:
        import numpy as np
        from pynwb import NWBHDF5IO
    except ImportError:
        print("WARNING: pynwb/numpy not available. Returning None.")
        return None

    nwb_file = Path(nwb_path)
    if not nwb_file.exists():
        print("  NWB file not found: {}".format(nwb_path))
        return None

    with NWBHDF5IO(str(nwb_file), 'r') as io:
        nwbfile = io.read()

        session_id = nwbfile.session_id or nwb_file.stem
        animal_id = nwbfile.subject.subject_id if nwbfile.subject else 'unknown'
        date = str(nwbfile.session_start_time.date()) if nwbfile.session_start_time else 'unknown'

        # Extract units
        units_table = nwbfile.units
        if units_table is None:
            print("  No units table in {}".format(nwb_path))
            return None

        ca1_units = []
        pfc_units = []
        all_unit_info = []

        for idx in range(len(units_table)):
            unit_meta = {}
            for col in units_table.colnames:
                try:
                    unit_meta[col] = units_table[col][idx]
                except Exception:
                    pass

            try:
                region, is_input = classify_unit_region(unit_meta)
            except ValueError:
                continue

            quality = str(unit_meta.get('quality', 'good'))
            if quality == 'noise':
                continue

            spike_times = units_table['spike_times'][idx]
            mean_fr = len(spike_times) / max(spike_times[-1] - spike_times[0], 1e-6) if len(spike_times) > 1 else 0.0

            ui = UnitInfo(
                unit_id='unit_{}'.format(idx),
                brain_region=region,
                electrode_id=str(unit_meta.get('electrode', idx)),
                quality=quality,
                mean_firing_rate=round(mean_fr, 2),
                is_input=is_input,
            )
            all_unit_info.append(ui)

            if is_input:
                ca1_units.append((idx, spike_times))
            else:
                pfc_units.append((idx, spike_times))

        if not ca1_units or not pfc_units:
            print("  Skipping {}: need both CA1 and PFC units".format(session_id))
            return None

        # Determine time range and bins
        all_times = []
        for _, st in ca1_units + pfc_units:
            if len(st) > 0:
                all_times.extend([st[0], st[-1]])
        t_start = min(all_times)
        t_end = max(all_times)
        time_bins_arr = np.arange(t_start, t_end + bin_size_s, bin_size_s)

        # Bin spikes
        n_bins = len(time_bins_arr) - 1
        ca1_matrix = np.zeros((n_bins, len(ca1_units)), dtype=np.float32)
        pfc_matrix = np.zeros((n_bins, len(pfc_units)), dtype=np.float32)

        for col_idx, (_, st) in enumerate(ca1_units):
            ca1_matrix[:, col_idx] = bin_spikes(
                list(st), list(time_bins_arr))

        for col_idx, (_, st) in enumerate(pfc_units):
            pfc_matrix[:, col_idx] = bin_spikes(
                list(st), list(time_bins_arr))

        # Extract LFP if available
        lfp_ca1 = None
        lfp_pfc = None
        lfp_sr = 1250.0  # typical LFP rate

        # Extract position if available
        position = None
        if nwbfile.processing and 'behavior' in nwbfile.processing:
            behavior = nwbfile.processing['behavior']
            if 'Position' in behavior.data_interfaces:
                pos_interface = behavior.data_interfaces['Position']
                for ss in pos_interface.spatial_series.values():
                    position = np.array(ss.data)
                    break

        # Trial info
        trials = nwbfile.trials
        trial_starts = None
        trial_ends = None
        n_trials = 0
        if trials is not None:
            n_trials = len(trials)
            trial_starts = np.array(trials['start_time'][:])
            trial_ends = np.array(trials['stop_time'][:])

        session_info = SessionInfo(
            session_id=session_id,
            animal_id=animal_id,
            date=date,
            n_ca1_units=len(ca1_units),
            n_pfc_units=len(pfc_units),
            n_trials=n_trials,
            duration_s=round(t_end - t_start, 2),
            sampling_rate_spikes=1.0 / bin_size_s,
            sampling_rate_lfp=lfp_sr,
        )

        return SessionData(
            session_info=session_info,
            units=all_unit_info,
            ca1_spikes=ca1_matrix,
            pfc_spikes=pfc_matrix,
            lfp_ca1=lfp_ca1,
            lfp_pfc=lfp_pfc,
            position=position,
            time_bins=time_bins_arr[:-1],
            trial_starts=trial_starts,
            trial_ends=trial_ends,
        )


# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------

def load_dandi000978(data_dir: str) -> DANDI000978Dataset:
    """Load all sessions from DANDI 000978.

    Parameters
    ----------
    data_dir : str
        Path to local directory containing downloaded NWB files.
        Expected structure:
            data_dir/
                sub-XX/
                    sub-XX_ses-YY.nwb
                OR
                *.nwb (flat structure)

    Returns
    -------
    DANDI000978Dataset
    """
    dataset = DANDI000978Dataset()
    data_path = Path(data_dir)

    if not data_path.exists():
        print("WARNING: Data directory '{}' not found. "
              "Returning empty dataset.".format(data_dir))
        return dataset

    # Find all NWB files
    nwb_files = sorted(data_path.rglob("*.nwb"))
    if not nwb_files:
        print("WARNING: No .nwb files found in {}".format(data_dir))
        return dataset

    print("Found {} NWB files in {}".format(len(nwb_files), data_dir))

    for nwb_path in nwb_files:
        print("  Loading: {}".format(nwb_path.name))
        session = load_nwb_session(str(nwb_path))
        if session is not None:
            dataset.sessions.append(session)

    dataset.n_sessions = len(dataset.sessions)
    return dataset


# ---------------------------------------------------------------------------
# Convert to unified HDF5
# ---------------------------------------------------------------------------

def convert_to_unified_hdf5(dataset: DANDI000978Dataset,
                            output_path: str) -> str:
    """Convert DANDI000978Dataset to unified HDF5.

    Output structure:
        /session_XX/
            ca1_spikes      (n_timesteps, n_ca1)
            pfc_spikes      (n_timesteps, n_pfc)
            lfp_ca1         (n_lfp, n_ca1_ch)    if available
            lfp_pfc         (n_lfp, n_pfc_ch)    if available
            position        (n_pos, 2)            if available
            time_bins       (n_timesteps,)
            trial_starts    (n_trials,)
            trial_ends      (n_trials,)
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
        hf.attrs['dandi_id'] = dataset.dandi_id
        hf.attrs['input_region'] = dataset.input_region
        hf.attrs['output_region'] = dataset.output_region
        hf.attrs['species'] = dataset.species
        hf.attrs['task'] = dataset.task
        hf.attrs['n_sessions'] = dataset.n_sessions

        for session in dataset.sessions:
            sid = session.session_info.session_id
            grp = hf.create_group(sid)

            # Session metadata
            for fld in ('animal_id', 'date', 'n_ca1_units', 'n_pfc_units',
                        'n_trials', 'duration_s', 'sampling_rate_spikes',
                        'sampling_rate_lfp', 'maze_type'):
                grp.attrs[fld] = getattr(session.session_info, fld)

            # Spike matrices
            if session.ca1_spikes is not None:
                grp.create_dataset('ca1_spikes', data=session.ca1_spikes)
            if session.pfc_spikes is not None:
                grp.create_dataset('pfc_spikes', data=session.pfc_spikes)

            # LFP
            if session.lfp_ca1 is not None:
                grp.create_dataset('lfp_ca1', data=session.lfp_ca1)
            if session.lfp_pfc is not None:
                grp.create_dataset('lfp_pfc', data=session.lfp_pfc)

            # Position
            if session.position is not None:
                grp.create_dataset('position', data=session.position)

            # Time bins
            if session.time_bins is not None:
                grp.create_dataset('time_bins', data=session.time_bins)

            # Trial info
            if session.trial_starts is not None:
                grp.create_dataset('trial_starts', data=session.trial_starts)
            if session.trial_ends is not None:
                grp.create_dataset('trial_ends', data=session.trial_ends)

            # Unit metadata
            unit_info = [asdict(u) for u in session.units]
            grp.attrs['unit_info'] = json.dumps(unit_info)

    print("Unified HDF5 saved: {}".format(out_path))
    return str(out_path)


def _save_metadata_json(dataset: DANDI000978Dataset,
                        output_path: str) -> str:
    """Fallback JSON save when h5py unavailable."""
    out_path = Path(output_path).with_suffix('.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        'dandi_id': dataset.dandi_id,
        'input_region': dataset.input_region,
        'output_region': dataset.output_region,
        'species': dataset.species,
        'task': dataset.task,
        'n_sessions': dataset.n_sessions,
        'sessions': [],
    }
    for s in dataset.sessions:
        smeta = asdict(s.session_info)
        smeta['n_units'] = len(s.units)
        meta['sessions'].append(smeta)

    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("Metadata JSON saved: {}".format(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_dataset_summary(dataset: DANDI000978Dataset):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("DANDI 000978 — RAT CA1 + PFC W-MAZE")
    print("Input: {}  |  Output: {}".format(
        dataset.input_region, dataset.output_region))
    print("=" * 70)
    print("{:<20} | {:>6} | {:>6} | {:>6} | {:>8} | {}".format(
        "Session", "CA1", "PFC", "Trials", "Dur (s)", "Animal"))
    print("-" * 70)
    for s in dataset.sessions:
        si = s.session_info
        print("{:<20} | {:>6} | {:>6} | {:>6} | {:>8.1f} | {}".format(
            si.session_id, si.n_ca1_units, si.n_pfc_units,
            si.n_trials, si.duration_s, si.animal_id))
    print("-" * 70)
    print("Total sessions: {}".format(dataset.n_sessions))
    total_ca1 = sum(s.session_info.n_ca1_units for s in dataset.sessions)
    total_pfc = sum(s.session_info.n_pfc_units for s in dataset.sessions)
    print("Total CA1 units: {}  |  Total PFC units: {}".format(
        total_ca1, total_pfc))
    print("=" * 70)


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dandi_000978"
    ds = load_dandi000978(data_dir)
    print_dataset_summary(ds)
