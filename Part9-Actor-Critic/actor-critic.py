import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        # 通常，它会使用 softmax 函数将神经网络的原始输出（logits）转化为一组在 [0, 1] 之间的概率值，且概率总和为 1
        probs = self.actor(state)

        # 创建了一个离散概率分布对象，方便后续的 sample 操作
        # 基于给定的动作概率构造的多项分布。torch.distributions.Categorical 是 PyTorch 提供的概率分布工具，用于在强化学习中对离散动作进行采样
        action_dist = torch.distributions.Categorical(probs)

        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 将当前获取的数据进行转换，然后输入到网络中 
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device) # view(-1, 1) 确保张量形状为二维列向量，方便后续计算
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) 
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # 时序差分误差，代表当前Critic网络预测价值与实际价值目标之间的差值
        td_delta = td_target - self.critic(states)  

        # 使用Actor网络输出当前状态下各动作的概率，.gather(1, actions) 提取所选动作对应的概率值，再对其取对数，以便用于优化
        # 对数概率表示该动作被选择的“置信度”，便于后续的梯度更新
        log_probs = torch.log(self.actor(states).gather(1, actions)) 

        # Actor 策略网络的优化目标
        # 乘以-1 目的是取反，当TD误差为正（比预计更好），则增加该动作在对应状态的概率；当TD误差为负（比预计更差），则减少对应动作的概率
        # td_delta.detach() 防止td_delta对critic网络产生影响，确保critic网络更新与actor网络更新独立分开
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # Critic 价值网络的优化目标
        # 最小化预测值（状态价值）与TD目标的均方误差（MSE）
        # 使用detach()让梯度来自于Critic自身，不影响到其他地方的梯度计算
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # 在每轮优化前，清除原有梯度，避免梯度累积导致错误更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # 根据梯度信息，通过优化器（如Adam）进行参数更新，使网络的预测能力逐渐接近真实值
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
# plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
# plt.show()