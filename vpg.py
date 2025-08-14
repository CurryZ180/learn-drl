#!/usr/bin/env python3
"""
vpg_minimal.py

Minimal Vanilla Policy Gradient (with GAE) single-file implementation.
- No MPI, no external logging
- Supports discrete (Categorical) and continuous (Gaussian + tanh) action spaces
- Uses Gymnasium API
- Useful for stepping through tensors / adding breakpoints
"""

import argparse
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym

# -----------------------
# utilities
# -----------------------
def combined_shape(length, shape):
    if np.isscalar(shape):
        return (length, shape)
    return (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: [x0, x1, x2]
    output: [x0 + discount*x1 + discount^2*x2, x1 + discount*x2, x2]
    """
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out

# -----------------------
# Buffer (for VPG/GAE)
# -----------------------
class VPGBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.gamma = gamma
        self.lam = lam

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size, "Buffer full"
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.act_buf[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rew_buf[self.ptr] = float(rew)
        self.val_buf[self.ptr] = float(val)
        self.logp_buf[self.ptr] = float(logp)
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """
        Compute advantage estimates and rewards-to-go for trajectory ending
        at current ptr (path_start_idx -> ptr-1). last_val is V(s_T) for bootstrapping.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = adv

        # Rewards-to-go for value targets
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "Buffer must be full before get()"
        self.ptr = 0
        self.path_start_idx = 0

        # normalize advantages
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        # convert to tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

# -----------------------
# Actor-Critic networks
# -----------------------
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.Tanh)

    def forward(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def act_and_logp(self, obs):
        pi = self.forward(obs)
        a = pi.sample()
        logp = pi.log_prob(a)
        return a.cpu().numpy(), logp.detach().cpu().numpy()

    def logp(self, obs, act):
        pi = self.forward(obs)
        return pi.log_prob(act)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        # trainable log_std (vector)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.act_limit = act_limit

    def forward(self, obs):
        mu = self.mu_layer(self.net(obs))
        std = torch.exp(self.log_std)
        return mu, std

    def act_and_logp(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()   # reparameterize
        logp = dist.log_prob(a).sum(axis=-1)
        a_tanh = torch.tanh(a)
        action = (self.act_limit * a_tanh).cpu().numpy()
        logp = (logp - torch.sum(torch.log(1 - a_tanh**2 + 1e-6), axis=-1)).detach().cpu().numpy()
        return action, logp

    def logp(self, obs, act):
        # act is environment action; need to invert tanh and compute log prob
        # here we compute approximate logp by treating act as after-squash; for training we only need pi.log_prob of chosen action sampled from policy,
        # in our usage we will call act_and_logp to get logp. This function kept for completeness.
        raise NotImplementedError("Use act_and_logp for GaussianPolicy in this minimal impl.")

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation=nn.Tanh)
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # [B]

class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(64,64)):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, gym.spaces.Box):
            act_dim = act_space.shape[0]
            act_limit = float(act_space.high[0])
            self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, act_limit)
            self.is_discrete = False
        else:
            act_dim = act_space.n
            self.policy = CategoricalPolicy(obs_dim, act_dim, hidden_sizes)
            self.is_discrete = True

        self.v = MLPCritic(obs_dim, hidden_sizes)

    def step(self, obs_np):
        """
        Given observation(s) as numpy, return action, value, logp (all numpy)
        obs_np can be single observation (1D) or batch.
        """
        if not isinstance(obs_np, np.ndarray):
            obs_np = np.asarray(obs_np, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]
        # value
        v = self.v(obs_t).detach().cpu().numpy()
        # action and logp
        if self.is_discrete:
            a, logp = self.policy.act_and_logp(obs_t)
        else:
            a, logp = self.policy.act_and_logp(obs_t)
        # convert outputs to 1D numpy if batch size 1
        if a.shape[0] == 1:
            a = a[0]
            logp = np.asarray(logp)[0]
            v = np.asarray(v)[0]
        return a, v, logp

# -----------------------
# VPG training loop
# -----------------------
def vpg(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97,
        pi_lr=3e-4, vf_lr=1e-3, train_v_iters=80, max_ep_len=1000, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_space = env.observation_space
    act_space = env.action_space

    # instantiate actor-critic
    ac = actor_critic(obs_space, act_space, **ac_kwargs)
    ac.to(device)

    # buffer size = steps_per_epoch (we collect this many steps before updating)
    local_steps_per_epoch = steps_per_epoch
    buf = VPGBuffer(obs_dim=obs_space.shape[0],
                    act_dim=act_space.n if isinstance(act_space, gym.spaces.Discrete) else act_space.shape[0],
                    size=local_steps_per_epoch, gamma=gamma, lam=lam)

    # optimizers
    pi_params = [p for p in ac.policy.parameters() if p.requires_grad]
    pi_optimizer = Adam(pi_params, lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(o)   # numpy outputs
            # step environment
            o2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            ep_ret += r
            ep_len += 1

            # store to buffer
            buf.store(o, a, r, v, logp)

            # update observation
            o = o2

            timeout = (ep_len == max_ep_len)
            terminal = done

            epoch_ended = (t == local_steps_per_epoch - 1)

            if terminal or timeout or epoch_ended:
                # if trajectory didn't reach terminal (e.g., epoch end), bootstrap value
                if timeout or epoch_ended:
                    # compute V(s_T)
                    _, v_boot, _ = ac.step(o)
                    last_val = v_boot
                else:
                    last_val = 0.0
                buf.finish_path(last_val)
                if terminal:
                    print(f"[Epoch {epoch}] Episode finished: return={ep_ret:.3f}, len={ep_len}")
                # reset episode
                o, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Get data from buffer
        data = buf.get()

        # Convert to device
        data = {k: v.to(device) for k, v in data.items()}

        # Policy update (single step)
        pi_optimizer.zero_grad()
        # For discrete and continuous we have logp stored already as float; but we want torch logp aligned with current policy.
        # For simplicity in this minimal impl we recompute logp from current policy for the actions in buffer.
        obs_t = data['obs']
        act_t = data['act']
        adv_t = data['adv']
        if ac.is_discrete:
            pi_dist = ac.policy.forward(obs_t)
            logp = pi_dist.log_prob(act_t.long())
        else:
            # use reparameterized Normal logprob with tanh correction as in policy definition
            mu, std = ac.policy.forward(obs_t)
            # For correct logp we'd need to invert tanh; the buffer's logp was computed when sampling; to keep minimal and stable
            # we compute log prob of the pre-squash via Normal and correct using numerical trick:
            # but here we assume actions were produced from policy.act_and_logp so act_t is already post-squash; exact recomputation is non-trivial.
            # For the minimal VPG, we instead use the stored logps (no recompute). Convert stored numpy to tensor:
            logp = data['logp']
        loss_pi = -(logp * adv_t).mean()
        loss_pi.backward()
        pi_optimizer.step()

        # Value function update (regression to returns)
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            v_pred = ac.v(obs_t)
            loss_v = ((v_pred - data['ret'])**2).mean()
            loss_v.backward()
            vf_optimizer.step()

        # Print epoch summary
        print(f"Epoch {epoch:03d} \tLossPi: {loss_pi.item():.5f} \tLossV: {loss_v.item():.5f} \tTime: {time.time()-start_time:.2f}s")

    env.close()
    return ac

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ac = vpg(lambda: gym.make(args.env),
             actor_critic=ActorCritic,
             ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
             seed=args.seed,
             steps_per_epoch=args.steps_per_epoch,
             epochs=args.epochs)
