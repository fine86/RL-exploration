import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class exploreEnv(object):
    def __init__(self, env, observe):
        self.env = env
        self.size = len(env)
        self.observe = observe
        self.done = False
        self.state = [5, 5]
        self.action = [[1, 0],
                       [1, 1],
                       [0, 1],
                       [-1, 1],
                       [-1, 0],
                       [-1, -1],
                       [0, -1],
                       [1, -1]]
        self.obstacle = []
        self.observable_area = pow(self.size, 2)
        self.recent_action = [-1, -1]

    def reset_env(self, num):
        self.env = np.zeros((self.size, self.size))
        self.state = [5, 5]
        self.set_agent()
        self.done = False

        for i in range(num):
            obstacle = [random.randrange(20, 80), random.randrange(20, 80)]
            obstacle_size = random.randrange(5, 10)
            exploration.set_obstacle(obstacle, obstacle_size)

        return env

    def set_agent(self):
        if self.env[self.state[1], self.state[0]] == 0.7:
            print("Collision Occurs.")
            self.done = True

            return

        elif self.env[self.state[1], self.state[0]] == 0:
            self.observable_area -= 1 
        
        self.env[self.state[1], self.state[0]] = 1

        self.done = False

    # 장애물 설정
    def set_obstacle(self, obstacle, l):
        self.obstacle.append([obstacle[0], obstacle[1], l])
        
        for i in range(obstacle[0] - l, obstacle[0] + l + 1):
            for j in range(obstacle[1] - l, obstacle[1] + l + 1):
                if pow(i, 2) + pow(j, 2) <= pow(l, 2):
                    self.env[j, i] = 0.7
                    self.observable_area -= 1

    def explore(self, a):
        self.env[self.state[1], self.state[0]] = 0.4
        next_action = self.action[a]
        self.state[0] += next_action[0]
        self.state[1] += next_action[1]
        self.set_agent()
        self.recent_action[0] = self.recent_action[1]
        self.recent_action[1] = a
        reward = self.observation()

        return self.env, reward, self.done

    # 이동한 위치에서의 reward 계산
    def observation(self):
        positive_reward=0
        negative_reward=0
        
        if self.done:
            negative_reward -= 20000
            return negative_reward
        
        for i in range(max([0, self.state[0] - self.observe]), min([self.size, self.state[0] + self.observe] + 1)):
            for j in range(max([0, self.state[1] - self.observe]), min([self.size, self.state[1] + self.observe] + 1)):
                if pow(i, 2) + pow(j, 2) <= pow(self.observe, 2):
                    if self.env[j, i] == 0:
                        positive_reward += 1
                        self.env[j, i] = 0.4
                        self.observable_area -= 1
                    
                    elif self.env[j, i] == 0.7:
                        negative_reward -= 2
        
        if self.observable_area==0:
            positive_reward += 1000
            self.done = True

        elif positive_reward==0:
            negative_reward -= self.observable_area * 5
            self.done = True

        return positive_reward + negative_reward
    
    # 여기 학습할 RL 모델 구현

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N,4,84,84] -> [N,16,20,20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)         # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 11 * 11
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, -1)


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N,4,84,84] -> [N,16,20,20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)         # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 11 * 11
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PPO:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            gamma=0.99,
            lmbda=0.96,
            K_epoch=5,
            eps_clip=0.1
    ):
        self.data = []
        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.K_epoch = K_epoch
        self.eps_clip = eps_clip

        self.actor = Actor(state_dim[0], action_dim)
        self.critic = Critic(state_dim[0])

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.buffer_size=0
    def put_data(self, transition):
        self.data.append(transition)
        return self.buffer_size

    def make_batch(self):
        s_lst, a_lst, r_lst, ns_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, ns, prob_a, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            ns_lst.append(ns)
            prob_a_lst.append(prob_a)
            done_lst.append(done)

        s = torch.FloatTensor(np.array(s_lst))
        a = torch.LongTensor(np.array(a_lst))
        r = torch.FloatTensor(np.array(r_lst))
        ns = torch.FloatTensor(np.array(ns_lst))
        done = torch.FloatTensor(np.array(done_lst))
        prob_a = torch.FloatTensor(np.array(prob_a_lst))

        self.data = []
        return s, a, r, ns, done, prob_a

    def select_action(self, x):

        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        prob = self.actor(x)

        return prob

    def update_model(self):

        s, a, r, ns, done, prob_a = map(lambda x: x.to(self.device), self.make_batch())

        a = a.view(-1, 1)
        r = r.view(-1, 1)
        done = done.view(-1, 1)
        prob_a = prob_a.view(-1, 1)

        for i in range(self.K_epoch):
            td_target = r + self.gamma * (1 - done) * self.critic(ns)
            delta = td_target - self.critic(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0

            # GAE
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.FloatTensor(advantage_lst).to(self.device)

            action_probs = self.actor(s)
            pi_a = action_probs.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b==exp(log(a)-log(b))

            critic_loss = F.smooth_l1_loss(self.critic(s), td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

env_size = 100
env = np.zeros((env_size, env_size))

exploration = exploreEnv(env, 8)

max_epoch = 20
max_step = int(1e6)
total_steps = 0
eval_interval = 5000
state_dim = (1, 100, 100)
obstacle_num = 2
action_dim = exploration.action.__len__()

agent = PPO(state_dim, action_dim)



while total_steps > max_step:
    exploration.reset_env(obstacle_num)

    action_probs = agent.select_action()

    