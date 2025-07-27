import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

'''
torch.nn.Module 是所有神经网络模块的基类。继承这个基类可以让我们利用 PyTorch 提供的各种功能，如自动微分、优化器等
'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # ReLU 是一种常用的激活函数，可以引入非线性，从而使网络能够学习更复杂的特征
        # 将输入 x 通过第一个全连接层，得到隐藏层的输出
        x = F.relu(self.fc1(x)) 

        # 对动作的原始分数应用 Softmax 激活函数，得到动作的概率分布。Softmax 函数将原始分数转换为概率，使得所有动作的概率和为1。
        # dim=1 表示在第二个维度上应用 Softmax 函数（即对每个样本的动作分数进行归一化）
        return F.softmax(self.fc2(x), dim=1) # 将隐藏层的输出通过第二个全连接层，得到动作的原始分数（logits）
    
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state): 
        # 将输入的状态转换为 PyTorch 张量，并移动到指定设备
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 根据当前状态生成动作概率分布，probs 是一个包含每个动作概率的张量
        probs = self.policy_net(state) 
        # 创建一个离散的概率分布对象 action_dist，其概率由 probs 指定
        action_dist = torch.distributions.Categorical(probs) 
        # 对动作概率分布进行随机采样
        action = action_dist.sample() 
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0

        # 在训练神经网络时，PyTorch 会自动计算损失函数相对于模型参数的梯度。这些梯度用于更新模型参数，以最小化损失函数。
        # 梯度是通过反向传播算法计算的，并存储在每个参数的 .grad 属性中
        # 在每次反向传播之前，必须将所有参数的梯度清零。这是因为在 PyTorch 中，梯度是累积的（即每次调用 .backward() 时，计算的梯度会累加到现有的梯度上）。
        # 如果不清零梯度，前一次计算的梯度会与当前计算的梯度相加，从而导致错误的梯度更新
        # 将模型中所有参数的梯度清零，以确保在进行新的梯度计算时，不受之前梯度的影响
        self.optimizer.zero_grad()
        # 反向遍历经验数据，从最后一步开始计算。这是因为在策略梯度方法中，回报是从当前时间步到回合结束的累积奖励
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action)) # 计算当前状态下选择动作的对数概率
            G = self.gamma * G + reward # 更新累积回报
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度,将梯度累积到模型参数的 .grad 属性中
        self.optimizer.step()  # 梯度下降

learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset(seed = 0)[0]
            done = False
            steps = 0
            while not done and steps < 200:
                steps += 1
                action = agent.take_action(state)
                next_state, reward, done, _, __ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('datas/Part8-Policy-based/REINFORCE_CartPole-v1.png')

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('datas/Part8-Policy-based/REINFORCE_CartPole-v1_moving_average.png')