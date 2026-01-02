# GWT-IB-SEC: Gated Working Memory with Information Bottleneck

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

A production-ready implementation of **GWT-IB** (Gated Working Memory + Information Bottleneck) using PPO for reinforcement learning on memory-intensive tasks. This implementation combines recurrent neural networks with information-theoretic regularization to achieve selective memory processing and improved generalization on partially observable sequential decision-making problems.

## Table of Contents

- [Overview](#overview)
- [Algorithm Theory](#algorithm-theory)
- [Architecture Details](#architecture-details)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Results and Analysis](#results-and-analysis)
- [Implementation Details](#implementation-details)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

GWT-IB combines three key innovations:

1. **Gated Working Memory (GWT)**: Selective information routing through learned sigmoid gates
2. **Information Bottleneck (IB)**: Compression regularization using KL-divergence constraints
3. **TD Statistics Normalization**: Stable temporal difference learning with online statistics

This approach addresses fundamental challenges in partially observable reinforcement learning:
- **Memory bottlenecks**: What information to store vs. discard
- **Credit assignment**: Linking distant observations to rewards
- **Generalization**: Learning robust representations across episodes

## Algorithm Theory

### Gated Working Memory (GWT)

The GWT mechanism implements a selective attention system that controls information flow through the recurrent network:

```
c_t = σ(W_router · [z_t, h_t-1, td_t-1] + b_router)
h_t = GRU(z_t * c_t, h_t-1)
```

Where:
- `z_t`: Encoded observation at time t
- `h_t`: Hidden state (working memory)
- `c_t`: Gate activation (0=block, 1=pass)
- `td_t-1`: Previous temporal difference error
- `σ`: Sigmoid activation

**Key Insight**: The gate `c_t` is conditioned on the previous TD error, allowing the model to adapt information filtering based on learning progress.

### Information Bottleneck (IB)

The information bottleneck principle constrains the mutual information between the gate activations and a prior distribution:

```
L_IB = D_KL(c_t || prior_t)
L_total = L_PPO + λ_c * L_IB
```

Where:
- `prior_t`: Target gate distribution (learned parameter)
- `λ_c`: Information bottleneck strength (scheduled)
- `D_KL`: Kullback-Leibler divergence

**Purpose**: Forces the model to compress information efficiently, preventing overfitting to irrelevant details while preserving task-relevant signals.

### TD Statistics Normalization

Traditional PPO can suffer from unstable value learning. Our approach normalizes TD errors using online statistics:

```
td_normalized = (td_raw - td_mean) / (td_std + ε)
td_clipped = clip(td_normalized, -td_clip, td_clip)
```

With exponential moving averages:
```
td_mean ← α * td_mean + (1-α) * td_current
td_std ← α * td_std + (1-α) * |td_current - td_mean|
```

**Benefits**: Stable learning across different reward scales and improved convergence.

## Architecture Details

### Network Structure

```
Input: Observation [obs_dim]
    ↓
Encoder: obs_dim → 64 → 64 [ReLU activation]
    ↓
Router: [z_t, h_t-1, td_t-1] → c_t [Sigmoid gate]
    ↓
Gated Input: z_t * c_t
    ↓
GRU: 64 hidden units
    ↓
Policy Head: 64 → action_dim [Categorical]
Value Head: 64 → 1 [Linear]
```

### Key Components

1. **Encoder**: 2-layer MLP with ReLU activations
   - Maps raw observations to feature space
   - Dimension: `obs_dim → 64 → 64`

2. **Router (Gate)**: Information bottleneck controller
   - Input: `[encoded_obs, prev_hidden, prev_td_error]`
   - Output: Gate probability `c_t ∈ [0,1]`
   - Learnable bias parameter for gate centering

3. **GRU**: Recurrent memory processor
   - 64 hidden units
   - Processes gated encoded observations
   - Maintains temporal state across episode

4. **Policy/Value Heads**: Standard PPO outputs
   - Policy: Categorical distribution over actions
   - Value: Scalar state-value estimation

### Memory Management

The model maintains several state variables:
- `h_t`: GRU hidden state (working memory)
- `td_prev`: Previous TD error (for gate conditioning)
- `done_prev`: Episode termination mask
- `td_stats`: Online mean/std for normalization

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- POPGym (memory task environments)
- Gymnasium (OpenAI Gym replacement)

### Quick Install

```bash
git clone https://github.com/dawsonblock/GWT-IB-SEC.git
cd GWT-IB-SEC
pip install popgym gymnasium torch tqdm matplotlib numpy
```

### Development Install

```bash
git clone https://github.com/dawsonblock/GWT-IB-SEC.git
cd GWT-IB-SEC
pip install -e .
pip install -r requirements-dev.txt  # If available
```

## Usage

### Basic Training

```bash
python gwt_ib_popgym_fixall.py
```

### Monitor Training

The script outputs real-time training metrics:

```
upd  100/244 | td_mean +0.000 td_std 0.022 | mean_c 0.306 | EV -0.000 | bias -0.023 | 3,502 steps/s
```

- `upd`: Update number / total updates
- `td_mean/std`: Temporal difference statistics
- `mean_c`: Average gate activation
- `EV`: Explained variance of value function
- `bias`: Learnable router bias
- `steps/s`: Training throughput

### Output Files

Training generates:
- Console logs with training metrics
- Final evaluation plots (TD→c correlation, performance curves)
- Model checkpoints (if implemented)

## Configuration

Key hyperparameters in the `Cfg` class:

### Environment Settings
```python
env_id = "popgym-CountRecallEasy-v0"  # POPGym task
num_envs = 4                          # Parallel environments
```

### Training Parameters
```python
total_timesteps = 1_000_000          # Total training steps
rollout_steps = 128                  # Steps per rollout
batch_size = 512                     # Minibatch size
lr = 2.5e-4                         # Learning rate
gamma = 0.99                        # Discount factor
```

### PPO Hyperparameters
```python
clip_eps = 0.2                      # PPO clipping
vf_coef = 0.5                       # Value function loss weight
ent_coef = 0.01                     # Entropy bonus
max_grad_norm = 0.5                 # Gradient clipping
```

### GWT-IB Specific
```python
hid_dim = 64                        # GRU hidden size
enc_dim = 64                        # Encoder dimension
lambda_c_start = 0.1                # Initial IB strength
lambda_c_end = 0.01                 # Final IB strength
td_clip = 5.0                       # TD error clipping
```

### TD Statistics
```python
td_ema_alpha = 0.99                 # EMA smoothing factor
td_quantile_low = 0.1               # Lower quantile for normalization
td_quantile_high = 0.9              # Upper quantile for normalization
```

## Training Process

### Phase 1: Rollout Collection
1. **Environment interaction**: Collect experiences using current policy
2. **Gate conditioning**: Compute gates based on `[obs, hidden, prev_td]`
3. **Information flow**: Gate controls what information passes to GRU
4. **TD calculation**: Compute temporal difference errors for next iteration
5. **Statistics update**: Online update of TD mean/std with EMA

### Phase 2: Policy Optimization
1. **Advantage estimation**: GAE with clean bootstrap (td_prev=0)
2. **Minibatch sampling**: Random sampling from collected rollouts
3. **PPO updates**: Clipped policy and value function optimization
4. **IB regularization**: KL divergence penalty on gate activations
5. **Router bias update**: Learnable parameter for gate centering

### Phase 3: Evaluation
- Periodic evaluation on fresh episodes
- Performance tracking and logging
- Gate activation analysis
- Correlation analysis between TD errors and gate activations

### Key Training Features

1. **No Gradient Tracking in Rollouts**: Rollout collection wrapped in `torch.no_grad()` to prevent double-backward errors

2. **Clean Bootstrap**: Next-state values computed with `td_prev=0` to avoid bias in TD target calculation

3. **Dtype Consistency**: All tensors explicitly cast to `float32` to prevent mixed-precision errors

4. **Stable TD Learning**: Quantile-based normalization prevents explosion of TD errors

## Results and Analysis

### Expected Training Behavior

**Healthy Training Indicators:**
- `td_mean` converging to ~0.0 (unbiased TD learning)
- `td_std` stabilizing below 0.1 (consistent value learning)
- `mean_c` in range [0.2, 0.6] (selective gating, not fully open/closed)
- `EV` (explained variance) gradually improving
- Stable training throughput ~3,000+ steps/s

**Warning Signs:**
- `td_mean` drifting far from 0 (biased value function)
- `td_std` > 0.2 (unstable learning)
- `mean_c` near 0 or 1 (gate saturation)
- `EV` remaining negative (poor value function)

### Memory Task Performance

On POPGym CountRecallEasy-v0:
- **Task**: Remember and recall sequences of discrete tokens
- **Challenge**: Requires selective memory of relevant vs. irrelevant information
- **Success metric**: Episode return improvement over time

### Interpretation of Metrics

1. **Gate Analysis**: 
   - Low `mean_c`: Model being overly selective, may miss important information
   - High `mean_c`: Model not compressing information effectively
   - Optimal range: 0.3-0.5 for balanced information flow

2. **TD Statistics**:
   - Stable `td_mean` ≈ 0: Value function learning without bias
   - Decreasing `td_std`: Improved value function accuracy
   - Large fluctuations: May need learning rate adjustment

3. **Information Bottleneck**:
   - `lambda_c` scheduling: Start high (0.1) for exploration, decay to low (0.01)
   - Router `bias`: Learned parameter that centers gate activations
   - Correlation plots: Show relationship between TD errors and gate decisions

## Implementation Details

### Technical Innovations

1. **Gradient Management**: Careful separation of rollout collection (no gradients) and optimization phases

2. **Memory State Handling**: Proper detachment and reattachment of recurrent states between rollouts

3. **Bootstrapping Strategy**: Clean TD targets using `td_prev=0` for next-state value computation

4. **Numerical Stability**: Quantile-based TD normalization prevents training instabilities

### Code Organization

```python
class GWTIB(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg):
        # Network architecture definition
        
    def forward(self, obs_seq, h_prev, td_prev, done_prev, force_c=None):
        # Forward pass with gate computation
        
    def update_td_stats(self, td_batch):
        # Online statistics update
        
class Rollout:
    # Experience storage for PPO updates
    
def main():
    # Main training loop
    # - Rollout collection
    # - PPO optimization 
    # - Evaluation and logging
```

### Performance Optimizations

- Vectorized environment processing (4 parallel envs)
- Efficient tensor operations with proper device management
- Gradient clipping for stable optimization
- Memory-efficient rollout storage

## Troubleshooting

### Common Issues

1. **RuntimeError: Trying to backward through the graph a second time**
   - **Solution**: Ensure rollout collection is wrapped in `torch.no_grad()`
   - **Cause**: Gradient tracking during experience collection

2. **RuntimeError: mat1 and mat2 must have the same dtype**
   - **Solution**: Add `dtype=torch.float32` to all tensor creations
   - **Cause**: Mixed precision between observations and model parameters

3. **TypeError: TransformObservation missing observation_space**
   - **Solution**: Remove `gym.wrappers.TransformObservation` wrapper
   - **Cause**: API changes in Gymnasium

4. **ModuleNotFoundError: No module named 'popgym'**
   - **Solution**: `pip install popgym`
   - **Cause**: Missing environment dependency

### Performance Issues

1. **Slow Training**:
   - Check GPU availability: `torch.cuda.is_available()`
   - Reduce batch size if memory limited
   - Increase `num_envs` for better parallelization

2. **Unstable Learning**:
   - Reduce learning rate (try 1e-4)
   - Increase `td_clip` threshold
   - Adjust `lambda_c` scheduling

3. **Poor Memory Task Performance**:
   - Increase `hid_dim` (GRU size)
   - Reduce information bottleneck strength
   - Longer training (more timesteps)

### Debug Mode

To enable verbose logging:
```python
# Add to main() function
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

### Theoretical Background

1. **Information Bottleneck Principle**:
   - Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method.

2. **Proximal Policy Optimization**:
   - Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms.

3. **Partially Observable RL**:
   - Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains.

### Implementation References

- **POPGym**: Morad, S., et al. (2023). POPGym: Benchmarking partially observable reinforcement learning.
- **Gymnasium**: Towers, M., et al. (2023). Gymnasium.

### Related Work

- **Memory-Augmented Networks**: Graves, A., et al. (2016). Hybrid computing using a neural network with dynamic external memory.
- **Gated Recurrent Units**: Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder.
- **Attention Mechanisms**: Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.

---

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gwt-ib-sec-2026,
  title={GWT-IB-SEC: Gated Working Memory with Information Bottleneck for Sequential Decision Making},
  author={Block, Dawson},
  year={2026},
  url={https://github.com/dawsonblock/GWT-IB-SEC}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

For questions or issues, please open a GitHub issue.
