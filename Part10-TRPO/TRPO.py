import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy

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


class TRPO:
    """ TRPO算法 """
    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        # 1. 计算策略目标
        ## a. 计算代理目标函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        ## b. 计算代理目标函数对策略参数的梯度
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        ## c. 将梯度展平并拼接成一个向量
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # 2. 用共轭梯度法计算 x = H^(-1)g
        ## 目标函数的梯度 obj_grad、状态 states、旧的动作分布 old_action_dists
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        # 3. 计算最大步长系数，确保更新步长不会超过 KL 散度的限制
        ## a. 计算黑塞矩阵和更新方向的乘积 Hd
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        ## b. 计算最大步长系数 max_coef，确保 KL 散度不超过预设的阈值 kl_constraint
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))

        # 4. 在更新方向上进行线性搜索，找到合适的步长
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)

        # 5. 用线性搜索找到的新的策略参数更新策略网络
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update(self, transition_dict):
        # 1. 数据预处理
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 2. 计算 TD 目标和 TD 误差
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # 3. 计算优势函数
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        ## a. 计算旧策略下的动作对数概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        ## b. 计算旧策略下的动作分布
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        # 4. 更新价值网络
        ## a. 计算价值网络的损失 critic_loss：当前状态价值 self.critic(states) 与 TD 目标 td_target 之间的均方误差
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        ## b. 清零价值网络优化器的梯度
        self.critic_optimizer.zero_grad()
        ## c. 反向传播计算梯度
        critic_loss.backward()
        ## d. 更新价值网络的参数
        self.critic_optimizer.step()

        # 5.更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)
        
num_episodes = 500
hidden_dim = 128

# 折扣因子，通常介于0和1之间，值越接近于1，表示代理在决策时考虑更长远的奖励
gamma = 0.98

# 在 GAE（广义优势估计，Generalized Advantage Estimation）中使用的参数，用于平衡偏差和方差。
# lmbda 的值也介于0和1之间，值越大，表示更多地使用未来的信息来估计优势，从而减少估计的偏差，但可能增加方差
lmbda = 0.95

# 价值网络的学习率
critic_lr = 1e-2 

# 限制新旧策略之间的最大 Kullback-Leibler（KL）散度。这是TRPO的核心思想之一，即在更新策略时，不让新策略偏离旧策略太远，以确保学习的稳定性
kl_constraint = 0.0005

# 在策略更新时用于线性搜索的缩放系数。这个参数用于在策略更新的过程中逐步缩减步幅，以找到一个适合的更新方向。alpha 决定了每次线性搜索中比例缩放的步长
alpha = 0.5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)

# 设置 PyTorch 的全局随机数生成器的种子
# 通过设置随机数生成器的种子，你可以确保每次运行代码时生成的随机数序列是相同的，从而使实验结果具有可重复性
# 使得训练过程中任何涉及随机性的过程（例如，策略中的随机动作选择、模型权重的初始化等）在每次运行时都保持一致
torch.manual_seed(0)

agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
             kl_constraint, alpha, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
# plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
# plt.show()

