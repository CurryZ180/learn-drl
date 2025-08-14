import numpy as np
import torch
import gymnasium as gym
import time
from torch.optim import Adam
from copy import deepcopy
import itertools

# ReplayBuffer 用于存储经验
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # 强制转换为numpy数组，确保维度正确
        self.obs_buf[self.ptr] = np.array(obs, dtype=np.float32)
        self.obs2_buf[self.ptr] = np.array(next_obs, dtype=np.float32)
        self.act_buf[self.ptr] = np.array(act, dtype=np.float32)
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


# 定义MLP Actor-Critic网络
class MLPActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.actor = MLP([obs_dim] + list(hidden_sizes) + [act_dim], torch.nn.Tanh)
        self.critic1 = MLP([obs_dim + act_dim] + list(hidden_sizes) + [1], torch.nn.Tanh)
        self.critic2 = MLP([obs_dim + act_dim] + list(hidden_sizes) + [1], torch.nn.Tanh)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs)

    def q1(self, obs, act):
        return self.critic1(torch.cat([obs, act], dim=-1))

    def q2(self, obs, act):
        return self.critic2(torch.cat([obs, act], dim=-1))


# 定义MLP结构
class MLP(torch.nn.Module):
    def __init__(self, sizes, activation, output_activation=torch.nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# TD3算法的实现
def td3(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建环境
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # 创建actor-critic模型及目标网络
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.critic1.parameters(), ac.critic2.parameters())
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            pi_targ = ac_targ.act(o2)
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.act(o))
        return -q1_pi.mean()

    pi_optimizer = Adam(ac.actor.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        if timer % policy_delay == 0:
            for p in q_params:
                p.requires_grad = False

            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            for p in q_params:
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset()[0], False, 0, 0
            while not (d or ep_len == max_ep_len):
                o, r, d, _, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            print(f"Test Episode Return: {ep_ret}, Length: {ep_len}")

    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset()[0], 0, 0

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d or ep_len == max_ep_len:
            o, ep_ret, ep_len = env.reset()[0], 0, 0

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print(f"Epoch {epoch} - Total Steps: {t}")
            test_agent()


if __name__ == '__main__':
    td3(lambda: gym.make('Pendulum-v1'), actor_critic=MLPActorCritic, ac_kwargs=dict(hidden_sizes=[256, 256]), epochs=50)
