# GWT-IB-SEC: Gated Working Memory with Information Bottleneck

A production-ready implementation of GWT-IB (Gated Working Memory + Information Bottleneck) using PPO for reinforcement learning on memory tasks.

## Overview

This repository contains a recurrent PPO implementation with:
- **Gated Working Memory (GWT)**: Selective information routing using learned gates
- **Information Bottleneck (IB)**: Compression regularization for better generalization
- **TD Statistics Normalization**: Stable temporal difference learning
- **Clean TD Bootstrap**: Proper value function estimation

## Files

- `gwt_ib_popgym_fixall.py` - Main training script (production-ready)
- `ib` - Original implementation file
- `README.md` - This file

## Features

- Recurrent GRU-based architecture with gated information flow
- PPO with clipped policy and value function optimization
- Online TD statistics tracking with EMA smoothing
- Information bottleneck regularization with adaptive λ_c scheduling
- Router bias learning with sigmoid gating
- Evaluation on POPGym memory tasks (CountRecallEasy-v0)

## Dependencies

```bash
pip install popgym gymnasium torch tqdm matplotlib numpy
```

## Usage

```bash
python gwt_ib_popgym_fixall.py
```

## Training Progress

The algorithm trains for 1M timesteps with:
- TD normalization using quantile-based scaling
- Clean bootstrap (td_prev=0) for next-state values
- Gate checking with TD std sanity verification
- Evaluation every 10 updates

## Architecture

- **Encoder**: 2-layer MLP (obs_dim → enc_dim → enc_dim)
- **GRU**: Recurrent layer for temporal processing
- **Router**: Sigmoid-gated information bottleneck
- **Policy Head**: Categorical distribution over actions
- **Value Head**: State value estimation

## Results

Training metrics include:
- TD mean/std for learning stability
- Mean gate activation for information flow
- Explained variance for value function quality
- Router bias evolution

Training typically achieves:
- Stable TD learning (mean ≈ 0, std < 0.1)
- Selective gating (mean_c ∈ [0.2, 0.6])
- Improving explained variance over time
