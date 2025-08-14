# ddpg_minimal.py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy
import gymnasium as gym
import argparse
import time

# -----------------------
# Utilities
# -----------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# -----------------------
# ReplayBuffer
# -----------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr]  = np.asarray(obs, dtype=np.float32)
        self.act_buf[self.ptr]  = np.asarray(act, dtype=np.float32)
        self.rew_buf[self.ptr]  = rew
        self.obs2_buf[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, device='cpu'):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

# -----------------------
# Actor-Critic (DDPG)
# -----------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.ReLU, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # obs: [B, obs_dim]
        return self.act_limit * self.net(obs)   # tanh outputs in [-1,1], scale to env

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.squeeze(self.q(x), -1)  # [B]

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256), lr_pi=1e-3, lr_q=1e-3, device='cpu'):
        self.device = device
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, act_limit).to(device)
        self.q  = Critic(obs_dim, act_dim, hidden_sizes).to(device)

        self.pi_target = deepcopy(self.pi).to(device)
        self.q_target  = deepcopy(self.q).to(device)
        for p in self.pi_target.parameters(): p.requires_grad = False
        for p in self.q_target.parameters(): p.requires_grad = False

        self.pi_optimizer = Adam(self.pi.parameters(), lr=lr_pi)
        self.q_optimizer  = Adam(self.q.parameters(), lr=lr_q)

    def act(self, obs, noise_scale=0.0):
        # obs: np.ndarray (obs_dim,)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)   # [1, obs_dim]
        with torch.no_grad():
            a = self.pi(obs_t)          # [1, act_dim]
        a = a.cpu().numpy()[0]
        a = a + noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.pi.act_limit, self.pi.act_limit)

    def update(self, data, gamma=0.99):
        obs, act, rew, obs2, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # Critic update
        with torch.no_grad():
            a2 = self.pi_target(obs2)
            q2_target = self.q_target(obs2, a2)
            target_q = rew + gamma * (1 - done) * q2_target
        q_val = self.q(obs, act)
        q_loss = ((q_val - target_q)**2).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Actor update (maximize Q, equivalently minimize -Q)
        pi_loss = -self.q(obs, self.pi(obs)).mean()
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        return q_loss.item(), pi_loss.item()

    def soft_update(self, tau):
        # theta_target = tau*theta_online + (1-tau)*theta_target (note: common convention is opposite; choose consistent)
        for p, p_targ in zip(self.pi.parameters(), self.pi_target.parameters()):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)
        for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)

# -----------------------
# Training loop
# -----------------------
def ddpg(
    env_fn,
    seed=0,
    hidden_sizes=(256,256),
    epochs=50,
    steps_per_epoch=4000,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,       # for target networks (use (1 - polyak) as tau in soft_update)
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    act_noise=0.1,
    max_ep_len=1000,
    device='cpu'
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = DDPGAgent(obs_dim, act_dim, act_limit, hidden_sizes, lr_pi=pi_lr, lr_q=q_lr, device=device)
    rb = ReplayBuffer(obs_dim, act_dim, replay_size)

    total_steps = steps_per_epoch * epochs
    o, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    # Small helper to sample batch only when buffer has enough samples
    def can_sample():
        return rb.size >= batch_size

    print("Start training. Env:", env.unwrapped.spec.id)
    for t in range(total_steps):
        if t < start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o, noise_scale=act_noise)

        o2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        real_done = False if ep_len+1 == max_ep_len else done

        rb.store(o, a, r, o2, float(real_done))

        o = o2
        ep_ret += r
        ep_len += 1

        # End of episode
        if done or (ep_len == max_ep_len):
            print(f"[Step {t}] Episode return: {ep_ret:.3f}  length: {ep_len}")
            o, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        # Update networks
        if (t >= update_after) and (t % update_every == 0):
            for j in range(update_every):
                if not can_sample():
                    break
                batch = rb.sample_batch(batch_size, device=device)
                q_loss, pi_loss = agent.update(batch, gamma=gamma)
                # Use polyak as weighting: do soft update with tau = 1 - polyak (so small tau)
                tau = 1 - polyak
                agent.soft_update(tau)

        # End of epoch evaluation
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            # run a few deterministic episodes
            test_returns = []
            for _ in range(5):
                o_test, _ = test_env.reset()
                ret_test = 0.0
                done_test = False
                step_test = 0
                while not done_test and step_test < max_ep_len:
                    a_test = agent.act(o_test, noise_scale=0.0)  # deterministic (noise_scale=0)
                    o_test, r_test, term_test, trunc_test, _ = test_env.step(a_test)
                    done_test = term_test or trunc_test
                    ret_test += r_test
                    step_test += 1
                test_returns.append(ret_test)
            print(f"=== Epoch {epoch}  EvalAverageReturn {np.mean(test_returns):.3f}  returns: {test_returns} ===")

    env.close()
    test_env.close()
    # return agent for later use
    return agent

# -----------------------
# Run CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    start = time.time()
    ddpg(lambda: gym.make(args.env), seed=args.seed, epochs=args.epochs,
         steps_per_epoch=args.steps_per_epoch, device=args.device)
    print("Total time:", time.time() - start)
