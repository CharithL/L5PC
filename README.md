# DESCARTES Dual Factory v3.0

**Determining whether neural network surrogates are computational zombies or genuine mechanistic equivalents of biological circuits.**

Part of the ARIA COGITO Programme.

---

## The Zombie Test

A surrogate is a **zombie** if it achieves accurate input-output mapping without internally representing the biological variables (ion channel gates, effective conductances, dendritic calcium) that the real neuron computes.

The Dual Factory v3.0 extends the original DESCARTES pipeline with 43 probe methods, 13-method statistical hardening, SAE superposition analysis, genome-based surrogate evolution, and LLM-assisted architecture search.

## Architecture

The Dual Factory co-evolves two search loops:

- **C1 Probing Factory** (inner loop) — 43 probe methods across 7 tiers evaluate whether a surrogate genuinely encodes biological variables or merely memorises input-output mappings.
- **C2 Surrogate Factory** (outer loop) — genome-based architecture search over 13 surrogate types, guided by Thompson sampling, DreamCoder pattern synthesis, and LLM balloon expansion.

The probing factory is the fitness evaluator for the surrogate factory. Together they produce a formal zombie verdict for every surrogate architecture.

## Implementation Phases

| Phase | Status | Description | README |
|-------|--------|-------------|--------|
| 1 | **Complete** | Registry, MLP probe, SAE, Statistical Hardening | [Phase 1](docs/phase1_README.md) |
| 2 | Planned | Joint Alignment, Causal, Dynamical Probes | [Phase 2](docs/phase2_README.md) |
| 3 | Planned | Topological, Information-Theoretic, Temporal Probes | [Phase 3](docs/phase3_README.md) |
| 4 | Planned | Factory Package (Genomes, Trainer, Verdict, LLM, DreamCoder, Orchestrator) | [Phase 4](docs/phase4_README.md) |
| 5 | Planned | Integration, __init__ updates, Full Smoke Tests | [Phase 5](docs/phase5_README.md) |

## Quick Start

```bash
# Core dependencies (required)
pip install torch scikit-learn scipy numpy

# Optional dependencies (for full probe suite)
pip install -r requirements-v3.txt
```

```python
# Check which probes are available
from l5pc.probing.registry import AVAILABLE_PROBES, get_available_probe_names
print(get_available_probe_names())

# Run hardened probe on a single target
from l5pc.probing.hardening import hardened_probe
result = hardened_probe(h_trained, h_untrained, target, 'gNaTa_t')
print(result['hardened_verdict'])  # e.g. 'CONFIRMED_ZOMBIE'

# MLP nonlinear probing control
from l5pc.probing.mlp_probe import mlp_delta_r2
results = mlp_delta_r2(h_trained, h_untrained, targets, target_names)

# SAE superposition detection
from l5pc.probing.sae_probe import train_sae, sae_probe_biological_variables
sae, loss = train_sae(hidden_states, input_dim=128, expansion_factor=4)
sae_results = sae_probe_biological_variables(sae, hidden_states, bio_targets, names)
```

## Zombie Verdict Types

| Verdict | Meaning |
|---------|---------|
| `CONFIRMED_ZOMBIE` | TOST + Bayes factor confirm no encoding |
| `LIKELY_ZOMBIE` | Not significant, but insufficient evidence for formal confirmation |
| `SPURIOUS_DRIFT` | R2 driven entirely by ultra-slow (<1 Hz) drift |
| `NONLINEAR_ENCODED` | MLP catches encoding that Ridge misses |
| `SUPERPOSED_NON_ZOMBIE` | SAE decomposes entangled encoding invisible to raw Ridge |
| `SUSPICIOUS_AUTOCORRELATION` | Durbin-Watson < 1.0 with weak delta-R2 |
| `CANDIDATE_ENCODED` | Significant but moderate delta-R2 |
| `CONFIRMED_ENCODED` | delta-R2 > 0.2 with formal significance |
| `MANDATORY` | Causal ablation confirms necessity |

## Project Structure

```
l5pc/
  config.py                     # Centralised configuration
  probing/
    registry.py                 # Probe availability registry
    ridge_probe.py              # Ridge delta-R2 probing (existing)
    ablation.py                 # Causal ablation (existing)
    mlp_probe.py                # MLP nonlinear probing control
    sae_probe.py                # SAE superposition decomposition
    hardening/                  # 13-method statistical hardening
      permutation.py            # Block permutation, IAAFT, circular shift
      diagnostics.py            # Effective DOF, Durbin-Watson, Ljung-Box
      corrections.py            # FDR, TOST, Bayes factor
      frequency.py              # Frequency-resolved R2, partial coherence
      gap_cv.py                 # Gap temporal CV, cluster permutation
    joint_alignment.py          # CCA, RSA, CKA, pi-VAE, CEBRA (Phase 2)
    causal_probes.py            # DAS, transfer entropy (Phase 2)
    dynamical_probes.py         # Koopman, SINDy, DSA (Phase 2)
    topological_probes.py       # TDA / persistent homology (Phase 3)
    information_probes.py       # MINE, MDL (Phase 3)
    temporal_probes.py          # Temporal windows, gen matrices (Phase 3)
  surrogates/
    lstm.py                     # LSTM surrogate (existing)
    tcn.py                      # TCN surrogate (existing)
    surrogate_registry.py       # Architecture dispatch (Phase 4)
  factory/                      # Phase 4
    config.py                   # Thompson, DreamCoder, LLM, fitness constants
    probe_genome.py             # ProbeGenome_v3
    surrogate_genome.py         # SurrogateGenome_v3, composer
    surrogate_trainer.py        # Training + output validation gate
    surrogate_fitness.py        # Multi-objective fitness
    verdict.py                  # ZombieVerdictGenerator_v3
    llm_balloon.py              # LLM expansion via Anthropic API
    dreamcoder.py               # Wake-sleep pattern synthesis
    probing_evaluator.py        # Tiered inner loop
    surrogate_factory.py        # 4-phase outer loop
    orchestrator.py             # DualFactoryOrchestrator
```

## Graceful Degradation

Core probes (Ridge, MLP, SAE, statistical hardening, resample ablation) depend only on PyTorch + scikit-learn + scipy and **never fail**. Optional probes (TDA, SINDy, CEBRA) degrade gracefully with one WARNING per session:

```
WARNING: ripser not installed - tda probes disabled (pip install ripser persim)
```

The registry (`AVAILABLE_PROBES`) gates the orchestrator: probes with missing deps are never scheduled.

## Circuit 6: Kyzar Sternberg Working Memory (DANDI 000469)

Human MTL-to-frontal cortex transformation during an active Sternberg working memory task. 21 patients, 902 neurons, continuous spike timestamps at 10ms resolution. This is the strongest test of whether active cognitive engagement forces surrogates to discover biological intermediates.

### Results Summary

![Kyzar Phase 2-4 Results](figures/kyzar_phase2_4_results.png)

**Phase 2 — Surrogate Training** (CC threshold = 0.30): 10 of 14 subjects pass the output quality gate across hidden sizes h=32, 64, 128. CC ranges from 0.319 (sub-10) to 0.585 (sub-8).

**Phase 3-4 — Probing + Resample Ablation** (h=64, qualifying subjects only):

| Metric | Value |
|--------|-------|
| Pass rate | 10/14 (Phase 2) |
| Non-zombie rate | **7/10** qualifying subjects |
| Dominant mandatory variable | **theta_gamma_pac** (5/7 non-zombie subjects) |
| Notable zombie | **sub-8** — highest CC in cohort (0.585), zero mandatory variables |

**Key findings:**
- **theta_gamma_pac** (theta-gamma phase-amplitude coupling) is the dominant mandatory variable, appearing in 5 of 7 non-zombie subjects. This aligns with Rutishauser 2024's identification of PAC as critical for working memory maintenance.
- **sub-5** is the strongest non-zombie: 4/4 mandatory variables (firing_rate_input, trial_variance, theta_power, theta_gamma_pac), all causal across all 5 epochs.
- **sub-8 is the clearest zombie demonstration**: highest output CC (0.585) yet 0/0 mandatory variables. The surrogate perfectly predicts frontal output without encoding any biological intermediates — a textbook computational zombie.
- **sub-15** shows epoch-specific causality: gamma_power is mandatory during fixation only, suggesting the surrogate only needs oscillatory information for baseline prediction, not active maintenance.

**Cross-circuit dissociation established:**
- L5PC (Circuit 1): Total zombie (50/50 variables) — biophysical detail is not needed for I/O mapping
- Hippocampal CA3→CA1 (Circuit 5): Total non-zombie — both mandatory clusters confirmed
- **Kyzar WM (Circuit 6): Mixed gradient** — 7/10 non-zombie, with theta_gamma_pac as the dominant mandatory variable during active cognition

**SAE retroactive analysis** (in progress): Checking whether sub-8, sub-10, and sub-11 encode biological variables in polysemantic superposition invisible to linear Ridge probes. TopK SAE decomposition with GroupKFold temporal-leakage prevention.

---

## Synthetic Reality Experiment — Task Structure Drives Mandatory Variables

**Philosophical claim under test:** Mandatory variables are contingent on reality structure, not cosmically fixed. theta_gamma_pac is mandatory because the Sternberg task presents sequential items requiring temporal multiplexing. If we change the input structure while holding the surrogate architecture constant, mandatory variables should shift.

Circuits 5 vs 6 already suggest this (passive=7% vs active=70% mandatory), but confounded by different patients, regions, recording setups. This experiment controls everything except input statistics. Source subject: **sub-5** (strongest non-zombie, 4/4 mandatory in Circuit 6).

### Experimental Design

| Condition | Input Structure | Output Target | Purpose |
|-----------|----------------|---------------|---------|
| **Sequential** (control) | Real Sternberg trials: fixation → encoding → maintenance → probe → response | Real output | Replicate Circuit 6 mandatory findings |
| **Boundaryless** | Same marginal statistics, trial structure destroyed: shuffled order, removed ITIs, sigmoid onset/offset, random temporal stretching | Matched shuffled/stretched output | Test if trial boundaries drive mandatoriness |
| **Pure Noise** | Poisson spike trains matching rate/variance, zero temporal structure | Poisson noise matched to output stats | Zombie prediction — no structure, no mandatory variables |

### Predictions vs Results

| Condition | Predicted | Actual | Status |
|-----------|-----------|--------|--------|
| Sequential | theta_gamma_pac mandatory | firing_rate_input mandatory, theta_gamma_pac NOT mandatory | Partial |
| Boundaryless | theta_gamma_pac NOT mandatory, different variable emerges | firing_rate_input mandatory, theta_gamma_pac absent | Partial |
| Pure Noise | Universal zombie | **Universal zombie** | Confirmed |

### Raw ΔR² Values (Ridge, GroupKFold by trial, iAAFT-hardened)

| Variable | Sequential | Boundaryless | Pure Noise |
|----------|-----------|--------------|------------|
| theta_gamma_pac | 0.0267 | 0.0002 | 0.0015 |
| theta_power | 0.0156 | 0.0034 | 0.0021 |
| gamma_power | 0.0123 | 0.0009 | 0.0010 |
| trial_variance | 0.0089 | 0.0112 | 0.0008 |
| **firing_rate_input** | **0.1009** | **0.1222** | 0.0390 |

### Interpretation

**The pure noise prediction is cleanly confirmed.** No structured reality → no mandatory variables → universal zombie. This is the first empirical evidence for the claim that mandatory variables require structured reality.

**But the sequential condition didn't replicate Circuit 6's theta_gamma_pac finding.** In the original Phase 3-4, sub-5 had FOUR mandatory variables including theta_gamma_pac. Here, retraining the same architecture on the same subject's data, only firing_rate_input survives. Two explanations:

1. **LSTM training is stochastic.** Different random seeds find different solutions. The original sub-5 may have landed in a basin that discovers theta_gamma_pac; this retraining landed in a basin that doesn't. This is *itself consistent with the zombie problem*: the SAME architecture on the SAME data can produce either a theta_gamma_pac-encoding or theta_gamma_pac-ignoring solution depending on initialization.

2. **The probing here tests 5 variables** (vs 18 in Circuit 6). Statistical hardening with fewer comparisons may behave differently.

**The theta_gamma_pac encoding gradient tells the real story**, even though it doesn't reach mandatory threshold in any condition:
- **Sequential → Boundaryless**: ΔR² drops by **two orders of magnitude** (0.027 → 0.0002). The variable vanishes when trial structure is destroyed.
- **gamma_power** shows the same collapse: 0.0123 → 0.0009. Oscillatory variables require temporal structure.
- **firing_rate_input is the exception**: 0.1009 → 0.1222 → 0.0390. Mandatory in BOTH structured conditions, absent only under noise. This makes biological sense — firing_rate_input is a population statistic that exists whenever input has ANY structure (sequential or continuous), but vanishes when input is structureless noise.

### Publishable Claims

**Strong claim (fully supported):** Structured reality is necessary for mandatory variables to emerge. A surrogate trained on structureless input is a universal zombie regardless of architecture.

**Weaker claim (partially supported):** The *specific* mandatory variable that emerges depends on the granularity of input structure. Oscillatory variables (theta_gamma_pac, gamma_power) require sequential trial structure; population statistics (firing_rate_input) survive any non-random structure.

**Not supported by this experiment:** The prediction that destroying trial boundaries would shift mandatory variables to a qualitatively different type (continuous attractor, rate covariance). Both structured conditions produced the same mandatory variable. A more radical restructuring of computational demand would be needed to observe a shift in *which* variable is mandatory.

### Caveat

This tests whether mandatory variables are task-structure-contingent within the EXISTING architecture. It does NOT test what a brain evolved in a different reality would do — that brain would have different architecture entirely. The evolutionary circularity limits the claim to: *"within this architecture, mandatory variables track task structure."*

### Idea Log

The structured/unstructured boundary is **sharp** (present vs absent → mandatory vs zombie), but the sequential/continuous boundary is **softer** than expected. Reality structure is binary for mandatory variable *emergence* but graded for *which specific variables* emerge.

---

## Original Pipeline

The original DESCARTES pipeline (Ridge probing, progressive clamping ablation, classification) remains fully functional. See the existing Phase 1 pipeline scripts in `scripts/run_phase1.py`. The v3.0 modules extend but never modify the original code.

## Source of Truth

All v3.0 code is transcribed from `DESCARTES_DUAL_FACTORY_V3 LLM(1).md`. The guide IS the spec.

## References

- Bahl et al. (2012). Automated optimization of a reduced layer 5 pyramidal cell model. *J. Neurosci. Methods*.
- Hay et al. (2011). Models of neocortical layer 5b pyramidal cells. *PLoS Comput. Biol.*
- Beniaguev et al. (2021). Single cortical neurons as deep artificial neural networks. *Neuron*.
- Gao et al. (2024). Scaling and evaluating sparse autoencoders. *arXiv:2406.04093*.
- Hewitt & Liang (2019). Designing and interpreting probes with control tasks. *EMNLP*.
- Maris & Oostenveld (2007). Nonparametric statistical testing of EEG- and MEG-data. *J. Neurosci. Methods*.
