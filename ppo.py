import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ActorCritic 网络定义（简单 MLP）
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64)):
        super().__init__()
        # 策略网络 π：输出动作概率分布参数（离散动作用 logits）
        layers = []
        last_size = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self.policy_net = nn.Sequential(*layers)
        self.logits = nn.Linear(last_size, act_dim)

        # 价值网络 V
        layers_v = []
        last_size = obs_dim
        for size in hidden_sizes:
            layers_v.append(nn.Linear(last_size, size))
            layers_v.append(nn.ReLU())
            last_size = size
        self.value_net = nn.Sequential(*layers_v)
        self.v = nn.Linear(last_size, 1)

    def step(self, obs):
        # obs: np.ndarray shape (obs_dim,)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape [1, obs_dim]
        # 计算动作概率分布
        logits = self.logits(self.policy_net(obs_t))
        pi_dist = torch.distributions.Categorical(logits=logits)
        a = pi_dist.sample()
        logp_a = pi_dist.log_prob(a)
        v = self.v(self.value_net(obs_t)).squeeze(-1)
        # 打印调试信息
        # print(f"step() obs_t shape: {obs_t.shape}")
        # print(f"step() logits shape: {logits.shape}")
        # print(f"step() action: {a.item()}, logp: {logp_a.item()}, value: {v.item()}")
        return a.item(), v.item(), logp_a.item()

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.logits(self.policy_net(obs_t))
        pi_dist = torch.distributions.Categorical(logits=logits)
        a = pi_dist.sample()
        return a.item()

    def pi(self, obs, act=None):
        # 计算 pi 分布和 logp(动作)
        logits = self.logits(self.policy_net(obs))
        pi_dist = torch.distributions.Categorical(logits=logits)
        if act is None:
            return pi_dist, None
        logp_a = pi_dist.log_prob(act)
        return pi_dist, logp_a

    def v_func(self, obs):
        return self.v(self.value_net(obs)).squeeze(-1)

# 简单经验缓冲区，只存一步的数据（方便演示，没做复杂buffer）
class PPOBuffer:
    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = adv

        # Rewards-to-go for targets
        ret = discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = ret

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
                    act=torch.as_tensor(self.act_buf, dtype=torch.int64),
                    ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
                    adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
                    logp=torch.as_tensor(self.logp_buf, dtype=torch.float32))
        return data

def discount_cumsum(x, discount):
    """
    计算折扣累计和，方便计算优势和return
    """
    out = np.zeros_like(x)
    running_sum = 0
    for i in reversed(range(len(x))):
        running_sum = x[i] + discount * running_sum
        out[i] = running_sum
    return out

def ppo_cartpole():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac = MLPActorCritic(obs_dim, act_dim)
    pi_optimizer = Adam(ac.parameters(), lr=3e-4)
    vf_optimizer = Adam(ac.parameters(), lr=1e-3)

    steps_per_epoch = 4000
    epochs = 1000
    max_ep_len = 1000
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    train_pi_iters = 80
    train_v_iters = 80
    target_kl = 0.01

    buf = PPOBuffer(obs_dim, steps_per_epoch, gamma, lam)

    o, _ = env.reset()
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(o)

            o2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)

            o = o2

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(f"Warning: trajectory cut off by epoch at {ep_len} steps.")
                if timeout or epoch_ended:
                    _, v, _ = ac.step(o)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    print(f"Epoch {epoch} Episode Return: {ep_ret} Episode Length: {ep_len}")
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        data = buf.get()

        def compute_loss_pi(data):
            pi, logp = ac.pi(data['obs'], data['act'])
            ratio = torch.exp(logp - data['logp'])
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * data['adv']
            loss_pi = -(torch.min(ratio * data['adv'], clip_adv)).mean()
            approx_kl = (data['logp'] - logp).mean().item()
            ent = pi.entropy().mean().item()
            return loss_pi, approx_kl, ent

        def compute_loss_v(data):
            return ((ac.v_func(data['obs']) - data['ret']) ** 2).mean()

        # Policy update
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, kl, ent = compute_loss_pi(data)
            if kl > 1.5 * target_kl:
                print(f"Early stopping at step {i} due to reaching max KL.")
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Value function update
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        print(f"Epoch {epoch} finished. LossPi: {loss_pi.item():.4f}, LossV: {loss_v.item():.4f}, KL: {kl:.4f}, Entropy: {ent:.4f}")

if __name__ == '__main__':
    ppo_cartpole()
