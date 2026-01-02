"""
GWT-IB PRODUCTION-READY BUILD (FIX-ALL) — POPGYM CountRecallEasy-v0
===================================================================

What this fixes (without changing the router equation form):
1) TD normalization no longer pins to a large floor (td_std_min=1e-3, RMS/quantile scale).
2) Clean TD bootstrap (next-state value computed with td_prev=0 to avoid circular gating).
3) GateCheck includes td_std sanity so “perfect correlation but dead learning” can’t pass.

Install:
    pip install popgym gymnasium torch tqdm matplotlib

Run:
    python gwt_ib_popgym_fixall.py

Notes:
- POPGym envs are vector obs; we flatten to Box(obs_dim,).
- PPO is recurrent: we train on sequences (seq_len) with BPTT.
- Router equation preserved:
    c = sigmoid(td_scale * |td_norm| + base_logit + bias)

"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import popgym  # noqa: F401

from tqdm import trange
import matplotlib.pyplot as plt


# ============================================================
# 0) CONFIG
# ============================================================

@dataclass
class Cfg:
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Env
    env_id: str = "popgym-CountRecallEasy-v0"
    num_envs: int = 16
    seed: int = 42

    # Training
    total_timesteps: int = 500_000
    rollout_steps: int = 128
    seq_len: int = 32

    # PPO
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0

    # Model
    enc_dim: int = 128
    hid_dim: int = 256
    router_hid: int = 64

    # Router (equation shape preserved)
    td_scale: float = 4.0
    base_logit: float = -1.0
    bias_init: float = 0.0
    bias_clamp_min: float = -4.0
    bias_clamp_max: float = 2.0

    # Information bottleneck
    target_c: float = 0.35
    lambda_c: float = 0.02  # tuned for POPGym; start low
    budget_warmup_updates: int = 25

    # TD stats / normalization (FIXED)
    td_ema: float = 0.99
    td_scale_ema2: float = 0.995
    td_quantile: float = 0.90
    td_scale_mode: str = "ema_quantile"  # "ema" | "ema_quantile" | "quantile"
    td_center_mean: bool = True
    td_norm_clip: float = 5.0
    td_std_min: float = 1e-3
    td_std_max: float = 50.0
    use_percentile_buffer: bool = True
    td_buffer_size: int = 8192

    # Numeric safety
    td_clip: float = 20.0

    # Eval
    eval_every_updates: int = 20
    eval_episodes: int = 50

    # Ablations (optional)
    force_c: float | None = None  # set to 0.0 or 1.0 to force; None = normal


# ============================================================
# 1) UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_vec_env(cfg: Cfg):
    def thunk():
        env = gym.make(cfg.env_id)
        # POPGym uses dict obs sometimes; force flatten to 1D float
        env = gym.wrappers.FlattenObservation(env)
        return env

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(cfg.num_envs)])
    return envs


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # 1 - Var[y - yhat]/Var[y]
    var_y = torch.var(y_true)
    if var_y.item() < 1e-12:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-12))


# ============================================================
# 2) MODEL — GWT-IB (router eq form preserved)
# ============================================================

class GWTIB(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, cfg.enc_dim),
            nn.Tanh(),
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.Tanh(),
        )

        self.gru = nn.GRU(input_size=cfg.enc_dim, hidden_size=cfg.hid_dim)

        router_in = cfg.enc_dim + cfg.hid_dim
        self.router = nn.Sequential(
            nn.Linear(router_in, cfg.router_hid),
            nn.Tanh(),
            nn.Linear(cfg.router_hid, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
            nn.Tanh(),
            nn.Linear(cfg.hid_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
            nn.Tanh(),
            nn.Linear(cfg.hid_dim, 1),
        )

        # Learnable router bias (clamped)
        self.router_bias = nn.Parameter(torch.tensor([cfg.bias_init], dtype=torch.float32))

        # TD running stats (buffers)
        self.register_buffer("td_mean", torch.zeros(1))
        self.register_buffer("td_std", torch.ones(1))

        # Percentile buffer
        self.register_buffer("td_buffer", torch.zeros(cfg.td_buffer_size))
        self.register_buffer("td_buf_idx", torch.zeros(1, dtype=torch.long))
        self.register_buffer("td_buf_count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_td_stats(self, td: torch.Tensor):
        """
        FIXED: stable, non-pinched TD scale.
        - Tracks mean (optional)
        - Tracks magnitude via RMS and/or quantile with EMA
        - Writes buffer BEFORE quantile compute
        """
        cfg = self.cfg
        ema = float(cfg.td_ema)

        td_flat = td.detach().view(-1)
        if td_flat.numel() == 0:
            return

        batch_mean = td_flat.mean()
        batch_rms = torch.sqrt((td_flat * td_flat).mean().clamp(min=1e-12))

        # mean
        if cfg.td_center_mean:
            self.td_mean.mul_(ema).add_(batch_mean * (1 - ema))
        else:
            self.td_mean.mul_(ema)

        # percentile buffer write BEFORE quantile compute
        if cfg.use_percentile_buffer:
            td_abs = td_flat.abs()
            n = int(td_abs.numel())
            buf_size = int(cfg.td_buffer_size)
            idx = int(self.td_buf_idx.item())
            end = idx + n
            if end <= buf_size:
                self.td_buffer[idx:end] = td_abs
            else:
                first = buf_size - idx
                self.td_buffer[idx:] = td_abs[:first]
                self.td_buffer[: (end % buf_size)] = td_abs[first:]
            self.td_buf_idx.fill_(end % buf_size)
            new_count = int(self.td_buf_count.item()) + n
            self.td_buf_count.fill_(buf_size if new_count >= buf_size else new_count)

        def qscale(q: float):
            if cfg.use_percentile_buffer:
                count = int(self.td_buf_count.item())
                if count > 64:
                    data = self.td_buffer[:count] if count < int(cfg.td_buffer_size) else self.td_buffer
                    return torch.quantile(data, q)
            return torch.quantile(td_flat.abs(), q)

        # scale
        if cfg.td_scale_mode == "ema":
            target = batch_rms
            self.td_std.mul_(ema).add_(target * (1 - ema))
        elif cfg.td_scale_mode == "ema_quantile":
            target = qscale(float(cfg.td_quantile))
            ema2 = float(cfg.td_scale_ema2)
            self.td_std.mul_(ema2).add_(target * (1 - ema2))
        elif cfg.td_scale_mode == "quantile":
            self.td_std.copy_(qscale(float(cfg.td_quantile)))
        else:
            target = batch_rms
            self.td_std.mul_(ema).add_(target * (1 - ema))

        self.td_std.clamp_(min=float(cfg.td_std_min), max=float(cfg.td_std_max))

    def normalize_td(self, td: torch.Tensor) -> torch.Tensor:
        denom = self.td_std.clamp(min=self.cfg.td_std_min)
        td_norm = (td - self.td_mean) / denom
        return td_norm.clamp(-self.cfg.td_norm_clip, self.cfg.td_norm_clip)

    def forward(
        self,
        obs_seq: torch.Tensor,          # [T,B,obs_dim]
        h0: torch.Tensor,               # [1,B,hid_dim]
        td_prev_seq: torch.Tensor,      # [T,B,1]
        done_prev_seq: torch.Tensor,    # [T,B,1]
        force_c: float | None = None
    ):
        cfg = self.cfg
        T, B, _ = obs_seq.shape

        z = self.encoder(obs_seq)                       # [T,B,enc]
        h, hn = self.gru(z, h0)                         # [T,B,hid], [1,B,hid]

        # Router inputs
        router_in = torch.cat([z, h], dim=-1)           # [T,B,enc+hid]

        # TD conditioning (preserve form: sigmoid(scale*|td_norm| + base_logit + bias))
        td_norm = self.normalize_td(td_prev_seq)        # [T,B,1]
        td_abs_norm = td_norm.abs()

        base = cfg.base_logit
        bias = self.router_bias.clamp(cfg.bias_clamp_min, cfg.bias_clamp_max)
        prior = torch.sigmoid(cfg.td_scale * td_abs_norm + base + bias)  # [T,B,1]

        if force_c is None and cfg.force_c is not None:
            force_c = float(cfg.force_c)

        if force_c is not None:
            c = torch.full_like(prior, float(force_c))
        else:
            # router output is additive logit residual
            resid = self.router(router_in)              # [T,B,1]
            c = torch.sigmoid(torch.logit(prior.clamp(1e-5, 1 - 1e-5)) + resid)

        # Gate hidden state: write h_t <- c_t*h_t + (1-c_t)*stopgrad(h_t)
        # This keeps compute “selective” while allowing gradient through c into h.
        h_det = h.detach()
        h_gated = c * h + (1.0 - c) * h_det

        logits = self.actor(h_gated)                    # [T,B,act]
        value = self.critic(h_gated).squeeze(-1)        # [T,B]

        return {
            "logits": logits,
            "value": value,
            "hn": hn,
            "c": c.squeeze(-1),
            "prior": prior.squeeze(-1),
            "td_abs_norm": td_abs_norm.squeeze(-1),
            "bias": bias.detach().item(),
        }


# ============================================================
# 3) PPO STORAGE
# ============================================================

class Rollout:
    def __init__(self, T: int, B: int, obs_dim: int, device: str):
        self.T, self.B = T, B
        self.device = device
        self.obs = torch.zeros(T, B, obs_dim, device=device)
        self.actions = torch.zeros(T, B, dtype=torch.long, device=device)
        self.logp = torch.zeros(T, B, device=device)
        self.rewards = torch.zeros(T, B, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.values = torch.zeros(T, B, device=device)
        self.td_prev = torch.zeros(T, B, 1, device=device)

        # diagnostics (optional)
        self.c = torch.zeros(T, B, device=device)
        self.prior = torch.zeros(T, B, device=device)
        self.td_abs_norm = torch.zeros(T, B, device=device)

        # computed
        self.advantages = torch.zeros(T, B, device=device)
        self.returns = torch.zeros(T, B, device=device)

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        T = self.T
        adv = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values


# ============================================================
# 4) TRAIN / EVAL
# ============================================================

@torch.no_grad()
def evaluate(cfg: Cfg, model: GWTIB, env_id: str):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)

    model.eval()
    ep_returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed + 10_000)
        done = False
        h = torch.zeros(1, 1, cfg.hid_dim, device=cfg.device)
        td_prev = torch.zeros(1, 1, 1, device=cfg.device)
        done_prev = torch.zeros(1, 1, 1, device=cfg.device)

        ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, device=cfg.device, dtype=torch.float32).view(1, 1, -1)
            out = model(obs_t, h, td_prev, done_prev, force_c=cfg.force_c)
            logits = out["logits"][0, 0]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            # step
            obs2, r, term, trunc, _ = env.step(int(a.item()))
            done = bool(term or trunc)
            ret += float(r)

            # FIXED: clean bootstrap (td_prev=0) for td update
            obs2_t = torch.tensor(obs2, device=cfg.device, dtype=torch.float32).view(1, 1, -1)
            out2 = model(obs2_t, out["hn"], torch.zeros_like(td_prev), torch.tensor([[[float(done)]]], device=cfg.device), force_c=cfg.force_c)
            v = out["value"][0, 0]
            v2 = out2["value"][0, 0]
            td_curr = (r + cfg.gamma * float(1.0 - done) * float(v2.item()) - float(v.item()))
            td_curr = float(np.clip(td_curr, -cfg.td_clip, cfg.td_clip))

            # advance
            obs = obs2
            h = out["hn"]
            td_prev = torch.tensor([[[td_curr]]], device=cfg.device)
            done_prev = torch.tensor([[[float(done)]]], device=cfg.device)

        ep_returns.append(ret)

    model.train()
    return float(np.mean(ep_returns)), float(np.std(ep_returns))


def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    envs = make_vec_env(cfg)
    obs_dim = int(envs.single_observation_space.shape[0])
    act_dim = int(envs.single_action_space.n)

    model = GWTIB(obs_dim, act_dim, cfg).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # reset
    obs, _ = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, device=cfg.device, dtype=torch.float32)
    h = torch.zeros(1, cfg.num_envs, cfg.hid_dim, device=cfg.device)

    td_prev = torch.zeros(cfg.num_envs, 1, device=cfg.device)
    done_prev = torch.zeros(cfg.num_envs, 1, device=cfg.device)

    # stats
    returns_hist = deque(maxlen=100)
    updates = cfg.total_timesteps // (cfg.num_envs * cfg.rollout_steps)

    # diagnostics for plotting
    gatecheck_log = {"td_abs_norm": [], "c": [], "prior": [], "delta_c": []}
    eval_log = {"update": [], "mean": [], "std": []}

    global_step = 0
    t0 = time.time()

    for update in trange(updates, desc="updates"):
        # anneal lr
        frac = 1.0 - (update / max(1, updates))
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr * frac

        rollout = Rollout(cfg.rollout_steps, cfg.num_envs, obs_dim, cfg.device)

        # rollout collection
        model.eval()
        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                global_step += cfg.num_envs

                rollout.obs[t] = obs_t
                rollout.td_prev[t] = td_prev

                out = model(
                    obs_t.view(1, cfg.num_envs, obs_dim),
                    h,
                    td_prev.view(1, cfg.num_envs, 1),
                    done_prev.view(1, cfg.num_envs, 1),
                    force_c=cfg.force_c,
                )
                logits = out["logits"][0]           # [B,act]
                values = out["value"][0]            # [B]
                h_next = out["hn"]                  # [1,B,hid]

                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

                # step env
                obs2, r, term, trunc, _ = envs.step(actions.cpu().numpy())
                done = np.logical_or(term, trunc)

                # tensors
                obs2_t = torch.tensor(obs2, device=cfg.device, dtype=torch.float32)
                r_t = torch.tensor(r, device=cfg.device, dtype=torch.float32)
                done_t = torch.tensor(done.astype(np.float32), device=cfg.device)

                # FIXED: clean bootstrap for TD target (td_prev=0)
                out2 = model(
                    obs2_t.view(1, cfg.num_envs, obs_dim),
                    h_next,
                    torch.zeros(1, cfg.num_envs, 1, device=cfg.device),
                    done_t.view(1, cfg.num_envs, 1),
                    force_c=cfg.force_c,
                )
                next_v = out2["value"][0]

                td_curr = r_t + cfg.gamma * next_v * (1.0 - done_t) - values
                td_curr = td_curr.clamp(-cfg.td_clip, cfg.td_clip).view(cfg.num_envs, 1)

                # update TD stats online
                model.update_td_stats(td_curr)

                # store
                rollout.actions[t] = actions
                rollout.logp[t] = logp
                rollout.rewards[t] = r_t
                rollout.dones[t] = done_t
                rollout.values[t] = values

                # diagnostics
                rollout.c[t] = out["c"][0]
                rollout.prior[t] = out["prior"][0]
                rollout.td_abs_norm[t] = out["td_abs_norm"][0]

                # advance
                obs_t = obs2_t
                h = h_next
                td_prev = td_curr
                done_prev = done_t.view(cfg.num_envs, 1)

        # bootstrap value for GAE (clean)
        model.eval()
        with torch.no_grad():
            out_last = model(
                obs_t.view(1, cfg.num_envs, obs_dim),
                h,
                torch.zeros(1, cfg.num_envs, 1, device=cfg.device),
                done_prev.view(1, cfg.num_envs, 1),
                force_c=cfg.force_c,
            )
            last_value = out_last["value"][0]

        rollout.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # ==========================
        # PPO UPDATE (recurrent seq)
        # ==========================
        model.train()

        T, B = cfg.rollout_steps, cfg.num_envs
        assert T % cfg.seq_len == 0, "rollout_steps must be divisible by seq_len"
        n_seq = T // cfg.seq_len
        total_seqs = n_seq * B

        # reshape into sequences
        obs_seq = rollout.obs.view(n_seq, cfg.seq_len, B, obs_dim).permute(0, 2, 1, 3).reshape(total_seqs, cfg.seq_len, obs_dim)
        act_seq = rollout.actions.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len)
        logp_old_seq = rollout.logp.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len)
        adv_seq = rollout.advantages.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len)
        ret_seq = rollout.returns.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len)
        val_old_seq = rollout.values.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len)
        td_prev_seq = rollout.td_prev.view(n_seq, cfg.seq_len, B, 1).permute(0, 2, 1, 3).reshape(total_seqs, cfg.seq_len, 1)
        done_prev_seq = rollout.dones.view(n_seq, cfg.seq_len, B).permute(0, 2, 1).reshape(total_seqs, cfg.seq_len, 1)

        # normalize advantages per batch
        adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

        # warmup IB
        warm = min(1.0, (update + 1) / max(1, cfg.budget_warmup_updates))
        lambda_c = cfg.lambda_c * warm

        batch_size = total_seqs
        mb_size = batch_size // cfg.minibatches

        # gatecheck sampling (small)
        with torch.no_grad():
            td_abs_norm_flat = rollout.td_abs_norm.flatten().cpu().numpy()
            c_flat = rollout.c.flatten().cpu().numpy()
            prior_flat = rollout.prior.flatten().cpu().numpy()
            delta_c_flat = (c_flat - prior_flat)
            # store a small random slice
            take = min(4096, td_abs_norm_flat.shape[0])
            idx = np.random.choice(td_abs_norm_flat.shape[0], size=take, replace=False)
            gatecheck_log["td_abs_norm"].append(td_abs_norm_flat[idx])
            gatecheck_log["c"].append(c_flat[idx])
            gatecheck_log["prior"].append(prior_flat[idx])
            gatecheck_log["delta_c"].append(delta_c_flat[idx])

        for epoch in range(cfg.epochs):
            perm = torch.randperm(batch_size, device=cfg.device)
            for i in range(cfg.minibatches):
                mb_idx = perm[i * mb_size:(i + 1) * mb_size]

                obs_mb = obs_seq[mb_idx].transpose(0, 1)          # [L,MB,obs]
                act_mb = act_seq[mb_idx].transpose(0, 1)          # [L,MB]
                logp_old_mb = logp_old_seq[mb_idx].transpose(0, 1)
                adv_mb = adv_seq[mb_idx].transpose(0, 1)
                ret_mb = ret_seq[mb_idx].transpose(0, 1)
                val_old_mb = val_old_seq[mb_idx].transpose(0, 1)
                td_prev_mb = td_prev_seq[mb_idx].transpose(0, 1)  # [L,MB,1]
                done_prev_mb = done_prev_seq[mb_idx].transpose(0, 1)

                # init hidden per sequence = zeros (works fine for POPGym)
                h0 = torch.zeros(1, obs_mb.shape[1], cfg.hid_dim, device=cfg.device)

                out = model(obs_mb, h0, td_prev_mb, done_prev_mb, force_c=cfg.force_c)
                logits = out["logits"]                       # [L,MB,act]
                values = out["value"]                        # [L,MB]
                c = out["c"]                                 # [L,MB]
                prior = out["prior"]                          # [L,MB]

                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(logp - logp_old_mb)
                pg1 = -adv_mb * ratio
                pg2 = -adv_mb * torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss with clipping
                v_clipped = val_old_mb + torch.clamp(values - val_old_mb, -cfg.clip_eps, cfg.clip_eps)
                vf1 = (values - ret_mb) ** 2
                vf2 = (v_clipped - ret_mb) ** 2
                vf_loss = 0.5 * torch.max(vf1, vf2).mean()

                # IB bottleneck (per-sequence mean_c)
                # compute mean c per sequence element (MB dimension), then MSE to target
                mean_c_per_seq = c.mean(dim=0)                         # [MB]
                ib_loss = ((mean_c_per_seq - cfg.target_c) ** 2).mean()

                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy + lambda_c * ib_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

                # clamp bias after update (hard)
                with torch.no_grad():
                    model.router_bias.clamp_(cfg.bias_clamp_min, cfg.bias_clamp_max)

        # ==========================
        # Logging
        # ==========================
        with torch.no_grad():
            mean_c = float(rollout.c.mean().item())
            td_std = float(model.td_std.item())
            td_mean = float(model.td_mean.item())
            ev = explained_variance(rollout.values.flatten(), rollout.returns.flatten())

        if (update + 1) % cfg.eval_every_updates == 0 or update == 0:
            m, s = evaluate(cfg, model, cfg.env_id)
            eval_log["update"].append(update + 1)
            eval_log["mean"].append(m)
            eval_log["std"].append(s)

        if (update + 1) % 10 == 0:
            elapsed = time.time() - t0
            steps_per_s = global_step / max(1e-6, elapsed)
            print(
                f"upd {update+1:4d}/{updates} | "
                f"td_mean {td_mean:+.3f} td_std {td_std:.3f} | "
                f"mean_c {mean_c:.3f} | "
                f"EV {ev:.3f} | "
                f"bias {model.router_bias.item():+.3f} | "
                f"{steps_per_s:,.0f} steps/s"
            )

    # ============================================================
    # Final plots: TD->c and eval
    # ============================================================
    td_abs = np.concatenate(gatecheck_log["td_abs_norm"], axis=0)
    c_all = np.concatenate(gatecheck_log["c"], axis=0)
    prior_all = np.concatenate(gatecheck_log["prior"], axis=0)
    delta_all = np.concatenate(gatecheck_log["delta_c"], axis=0)

    # GateCheck: correlations
    def corr(a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt((a*a).mean()) * np.sqrt((b*b).mean()) + 1e-12)
        return float((a*b).mean() / denom)

    rho = corr(td_abs, c_all)
    rho_d = corr(td_abs, delta_all)

    # tail
    q = np.quantile(td_abs, 0.8)
    mask = td_abs >= q
    rho_tail = corr(td_abs[mask], c_all[mask]) if mask.any() else float("nan")
    rho_d_tail = corr(td_abs[mask], delta_all[mask]) if mask.any() else float("nan")

    # sanity: td_std not pinned
    std_ok = float(model.td_std.item()) > (cfg.td_std_min * 5.0)
    gate_status = "PASS" if (std_ok and (rho_tail > 0.2)) else "FAIL"

    print("\nGATECHECK")
    print(f"rho(td,c)={rho:.3f}  rho_tail={rho_tail:.3f} | rho(td,Δc)={rho_d:.3f} rho_tailΔ={rho_d_tail:.3f}")
    print(f"td_std={float(model.td_std.item()):.6f} (std_ok={std_ok}) -> {gate_status}")

    # plot TD->c binned
    bins = np.linspace(0.0, max(1e-6, np.quantile(td_abs, 0.99)), 25)
    inds = np.digitize(td_abs, bins) - 1
    bc, bp, bd, bx = [], [], [], []
    for k in range(len(bins) - 1):
        m = inds == k
        if m.sum() < 50:
            continue
        bx.append(0.5 * (bins[k] + bins[k + 1]))
        bc.append(c_all[m].mean())
        bp.append(prior_all[m].mean())
        bd.append(delta_all[m].mean())

    plt.figure()
    plt.plot(bx, bc, label="c (binned mean)")
    plt.plot(bx, bp, label="prior (binned mean)")
    plt.plot(bx, bd, label="Δc = c - prior (binned mean)")
    plt.xlabel("|td_norm|")
    plt.ylabel("gate")
    plt.title(f"TD→Gate | rho_tail={rho_tail:.3f}, td_std={float(model.td_std.item()):.4f}")
    plt.legend()
    plt.tight_layout()

    if len(eval_log["update"]) > 0:
        plt.figure()
        plt.plot(eval_log["update"], eval_log["mean"])
        plt.fill_between(
            eval_log["update"],
            np.array(eval_log["mean"]) - np.array(eval_log["std"]),
            np.array(eval_log["mean"]) + np.array(eval_log["std"]),
            alpha=0.2
        )
        plt.xlabel("update")
        plt.ylabel("eval return")
        plt.title("Evaluation")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()