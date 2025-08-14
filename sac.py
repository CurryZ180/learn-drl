# sac_minimal.py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
from copy import deepcopy

# ======================
# Utilities
# ======================
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# ======================
# Replay Buffer
# ======================
class ReplayBuffer:
    """Simple FIFO for SAC (continuous)."""
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, o, a, r, o2, d):
        self.obs_buf[self.ptr]  = np.asarray(o,  dtype=np.float32)
        self.obs2_buf[self.ptr] = np.asarray(o2, dtype=np.float32)
        self.act_buf[self.ptr]  = np.asarray(a,  dtype=np.float32)
        self.rew_buf[self.ptr]  = r
        self.done_buf[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256, device='cpu'):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

# ======================
# SAC Networks
# ======================
LOG_STD_MIN, LOG_STD_MAX = -20, 2

class SquashedGaussianMLPActor(nn.Module):
    """
    π(a|s): outputs tanh-squashed Gaussian with reparameterization trick.
    Returns (action, logp_pi) where gradients flow into action.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU, act_limit=1.0):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # obs: [B, obs_dim]
        net_out = self.net(obs)
        mu      = self.mu_layer(net_out)             # [B, act_dim]
        log_std = self.log_std_layer(net_out)        # [B, act_dim]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std     = torch.exp(log_std)

        # Reparameterization
        if deterministic:
            pi_action = mu
        else:
            noise = torch.randn_like(mu)
            pi_action = mu + noise * std

        # Squash by Tanh
        pi_action_squash = torch.tanh(pi_action)
        action = self.act_limit * pi_action_squash    # final action in env bounds

        if not with_logprob:
            return action, None

        # Compute logprob with Tanh correction
        # log N(mu, std) of pre-squash action
        pre_squash = pi_action
        logp = (-0.5 * (((pre_squash - mu) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))).sum(axis=-1)
        # Change of variables: log|det(d tanh(x) / dx)| = sum log(1 - tanh(x)^2)
        # Add small epsilon for numerical stability
        logp -= torch.sum(torch.log(1 - torch.tanh(pre_squash)**2 + 1e-6), dim=-1)
        return action, logp

class MLPQFunction(nn.Module):
    """Q(s,a): state and action are concatenated."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # [B]

class MLPActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        act_limit = float(act_space.high[0])

        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        # obs: np.ndarray (obs_dim,) or torch tensor [B, obs_dim]
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        a, _ = self.pi(obs, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

# ======================
# SAC Training Loop (Minimal)
# ======================
def sac(
    env_fn,
    hidden_sizes=(256, 256),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,              # fixed entropy coefficient (no auto-tuning for minimality)
    batch_size=256,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=5,
    max_ep_len=1000,
    device='cpu',
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    # Actor-Critic and target nets
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes).to(device)
    ac_targ = deepcopy(ac).to(device)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    # Optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_params = list(ac.q1.parameters()) + list(ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=lr)

    # Losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # Current Q estimates
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            # Next action from *current* policy
            a2, logp_a2 = ac.pi(o2)
            # Target Q-values from target nets
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        loss_q = ((q1 - backup)**2).mean() + ((q2 - backup)**2).mean()
        return loss_q

    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        # Entropy-regularized objective: maximize E[q - alpha*logp] => minimize alpha*logp - q
        loss_pi = (alpha * logp_pi - q_pi).mean()
        return loss_pi

    # Update step
    def update(data):
        # Q update
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q params for policy update to save compute
        for p in q_params:
            p.requires_grad = False

        # Policy update
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze
        for p in q_params:
            p.requires_grad = True

        # Polyak averaging: target ← ρ * target + (1-ρ) * online
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return float(loss_q.item()), float(loss_pi.item())

    # Evaluation
    @torch.no_grad()
    def test_agent():
        returns = []
        for _ in range(num_test_episodes):
            o, _ = test_env.reset()
            ep_ret, ep_len = 0, 0
            done = False
            while not done and ep_len < max_ep_len:
                a = ac.act(o, deterministic=True)
                o, r, terminated, truncated, _ = test_env.step(a)
                done = terminated or truncated
                ep_ret += r
                ep_len += 1
        returns.append(ep_ret)
        return np.mean(returns)

    # Main loop
    total_steps = steps_per_epoch * epochs
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0

    for t in range(total_steps):
        # Select action
        if t > start_steps:
            a = ac.act(o, deterministic=False)
        else:
            a = env.action_space.sample()

        # Step env
        o2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_ret += r
        ep_len += 1

        # Ignore done if timeout
        real_done = False if ep_len == max_ep_len else done

        # Store
        replay_buffer.store(o, a, r, o2, float(real_done))

        o = o2
        if done or (ep_len == max_ep_len):
            # episode end
            print(f"Step {t:7d}: EpRet={ep_ret:.2f}, EpLen={ep_len}")
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size, device=device)
                lq, lp = update(batch)

        # Eval per epoch
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            avg_ret = test_agent()
            print(f"[Epoch {epoch:03d}] EvalAverageReturn = {avg_ret:.2f}")

    env.close()
    test_env.close()

# ======================
# Run
# ======================
if __name__ == "__main__":
    # 默认用 Pendulum-v1（连续动作），避免 MuJoCo 依赖
    sac(lambda: gym.make("Pendulum-v1"),
        hidden_sizes=(256, 256),
        epochs=10,                # 可自行加大
        steps_per_epoch=4000,     # 可自行加大
        start_steps=1000,         # Pendulum 可以较小
        update_after=1000,
        batch_size=256,
        device="cpu")
