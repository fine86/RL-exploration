import numpy as np
import random
import math
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import wandb
from IPython.display import clear_output
from IPython.display import HTML

train_name = "[ver.3] mk2"

wandb.init(project = 'PPO Exploration Train')
wandb.run.name = train_name
wandb.save()

learning_rate=0.00025
gamma=0.99
lmbda=0.96
K_epoch=5
eps_clip=0.1

args = {
    "learning_rate" : learning_rate,
    "epochs" : K_epoch,
    "gamma" : gamma,
    "clip" : eps_clip,
    "lambda" : lmbda
}
wandb.config.update(args)

np.set_printoptions(threshold=np.inf, linewidth=np.inf) 

class exploreEnv(object):
    def __init__(self, env, observe, max_steps = 128):
        self.env = env
        self.steps = 0
        self.max_steps = max_steps
        self.size = len(env)
        self.observe = observe
        self.not_observed = 0
        self.done = False
        self.state = [10, 10]
        self.start = [10, 10]
        self.action = [[5, 0],
                       #[5, 5],
                       [0, 5],
                       #[-5, 5],
                       [-5, 0],
                       #[-5, -5],
                       [0, -5]]
                       #[5, -5]]
        self.next_action = [0, 0]
        self.obstacle = []
        self.observable_area = pow(self.size, 2)
        self.whole_area = self.observable_area
        self.recent_action = [-1, -1]
        self.negative_reward = 0
        self.positive_reward = 0
        self.observed_boundary = 0
        self.observed_step = 0
        self.final_observation = 0
        self.closest = [-1, -1]

    def reset_env(self, num):
        self.env = np.zeros((self.size, self.size))
        self.steps = 0
        self.state = [10, 10]
        self.start = [10, 10]
        self.set_agent()
        self.observable_area = pow(self.size, 2)
        self.whole_area = self.observable_area
        self.recent_observe = [-1, -1, -1, -1, -1]
        self.observation()
        self.done = False
        self.stop = False
        self.negative_reward = 0
        self.positive_reward = 0
        self.closest = [-1, -1]
        self.observed_boundary = 0
        self.observed_step = 0
        self.final_observation = 0

        for i in range(num):
            obstacle = [random.randrange(30, 70), random.randrange(30, 70)]
            obstacle_size = random.randrange(7, 10)
            self.set_obstacle(obstacle, obstacle_size)

        return self.env

    def set_agent(self):
        if self.state[0] < 0 or self.state[0] >= self.size or self.state[1] < 0 or self.state[1] >= self.size:
            # print("Error. Agent out of environment!")
            self.stop = True
            return

        if self.env[self.state[1], self.state[0]] == 0.7:
           # print("Collision Occurs.")
            self.stop = True
            return

        elif self.env[self.state[1], self.state[0]] == 0:
            self.observable_area -= 1 
        
        self.env[self.state[1], self.state[0]] = 1

        self.stop = False

    # 장애물 설정
    def set_obstacle(self, obstacle, l):
        self.obstacle.append([obstacle[0], obstacle[1], l])
        
        for i in range(obstacle[0] - l, obstacle[0] + l + 1):
            for j in range(obstacle[1] - l, obstacle[1] + l + 1):
                if pow(i - obstacle[0], 2) + pow(j - obstacle[1], 2) <= pow(l, 2):
                    self.env[j, i] = 0.7
                    self.observable_area -= 1
        
        self.whole_area = self.observable_area

    def explore(self, a):
        self.env[self.state[1], self.state[0]] = 0.4
        self.next_action = self.action[a]

        self.state[0] += self.next_action[0]
        self.state[1] += self.next_action[1]

        self.set_agent()
        self.steps += 1
        self.recent_action[0] = self.recent_action[1]
        self.recent_action[1] = a
        reward = self.observation()

        return self.env, reward, self.done
    
    def check_observed(self, distance):
        success = True
        for i in range(self.size):
            for j in range(self.size):
                if self.calc_dist([i, j], self.start) < distance - 20:
                    if self.env[j, i]==0:
                        success = False
                        break

        self.observed_boundary = distance
        self.observed_step = self.steps
        return success
    
    def calc_dist(self, base, target):
        distance = math.sqrt(pow(base[0] - target[0], 2) + pow(base[1] - target[1], 2))

        return distance

    # 이동한 위치에서의 reward 계산
    def observation(self):
        success = True
        observed_areas = 0
        observed_obstacles = 0
        
        for i in range(max(0, self.state[0] - self.observe), min(self.size, self.state[0] + self.observe + 1)):
            for j in range(max(0, self.state[1] - self.observe), min(self.size, self.state[1] + self.observe + 1)):
                if pow(i - self.state[0], 2) + pow(j - self.state[1], 2) <= pow(self.observe, 2):
                    if self.env[j, i] == 0:
                        observed_areas += 1
                        self.env[j, i] = 0.4
                        self.observable_area -= 1
                    
                    elif self.env[j, i] == 0.7:
                        observed_obstacles += 1
        
        current_distance = self.calc_dist(self.state, self.start)

        if current_distance < self.observed_boundary:
            self.negative_reward -= 100
        #2트
        ### 여기도 조건 추가해줘야 할 것 같음. step 간격을 기준으로 하면 안될 것 같은데
        #### 수정 완. step 간격을 기준으로 했음
        #### 대신 step 때문에 멈추면 negative reward 추가. max 600
        current_explore_step = self.steps - self.observed_step
        current_explore_boundary = current_distance - self.observed_boundary

        if current_explore_boundary > 20 or current_explore_step > 30:
            self.negative_reward -= current_explore_step * 20
            success = self.check_observed(current_distance)

        #2트
        ### 수정 완. 2000->3000
        if self.stop:
            self.negative_reward -= 3000
            self.done = True
        
        # 2트
        ### 수정 완. 1000->1500
        elif not success:
            self.negative_reward -= 1500
            self.done = True
        #2트
        ### 얘가 큰 듯. success만 계속하고 있으면 reward가 계속 커지니깐 50% 언저리에서 오락가락 하고 있는거 같음
        ### 근데 check 조건이 observed_boundary가 20 이상 커질 땐데 그게 되나?
        #### 수정 완. 100->30, max 128 * 20 = 2560
        elif success:
            self.positive_reward += 60

        elif self.observable_area == 0:
            self.positive_reward += 6000
            print("Success")
            self.done = True
        #2트
        ### 얘도 크게
        #### 수정 완. 50 -> 200
        elif observed_areas == 0:
            self.negative_reward -= 200
        #2트
        ### 얘는 크게
        #### 수정 완. 탐색한 거리의 제곱근에 비례하게
        ######## 만약에 다음에도 학습 안되면 1순위로 얘부터 고쳐야 함
        else:
            self.positive_reward += observed_areas * 0.3 * (math.sqrt(current_distance / (self.size * math.sqrt(2))) + 0.3)
        #2트
        ### 얘도 키워줘야 할 것 같음. total이 8000 언저린데 50% 학습한다고 더 멀리까지 찾지는 않을 듯?
        ### 얘도 더 많이 탐색할수록 비율 커지게 설정해야 할 듯(제곱?)
        ###### 수정 완료(제곱), positive 최대 5000
        ######### 안 될 시 경향성 보고 minimum 값 clip 하는 것도 고려
        if self.done or self.steps == self.max_steps:
            self.final_observation = 100 * pow((self.whole_area - self.observable_area) / self.whole_area + 0.3, 2)
            self.positive_reward += self.final_observation * 50

        return self.positive_reward + self.negative_reward

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
            lr=learning_rate,
            g=gamma,
            l=lmbda,
            e=K_epoch,
            clip=eps_clip
    ):
        self.data = []
        self.action_dim = action_dim
        self.gamma = g
        self.lmbda = l
        self.K_epoch = e
        self.eps_clip = clip

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
        #print('2')
        #print(s.size())
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

video_file = f'{train_name}.avi'
fps = 15
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(video_file, fourcc, fps, (100, 100), isColor = False)

print("GPU : ", torch.cuda.is_available())

env_size = 100
env = np.zeros((env_size, env_size))
#print(env)
exploration = exploreEnv(env, 15)
history = {'Step': [], 'AvgReturn': []}
horizon = 128
max_step = int(5e5)
max_score = -1e6
total_steps = 0
eval_interval = 5000
state_dim = (1, 100, 100)
obstacle_num = 0
action_dim = exploration.action.__len__()

agent = PPO(state_dim, action_dim)

def evaluate(n_evals=3, env_size=100, obstacle_num=0, eval_agent = agent):
    eval_env = np.zeros((env_size, env_size))
    eval_env = exploreEnv(eval_env, 15)

    scores = 0
    for i in range(n_evals):
        s = eval_env.reset_env(obstacle_num)
        done = eval_env.done
        while not done:
            action_probs = eval_agent.select_action(s)
            m = Categorical(action_probs)
            a = m.sample().item()
            s_prime, r, done = eval_env.explore(a)
            s = s_prime
        scores += r
    return np.round(scores / n_evals, 4)

def visualize(env_size=100, obstacle_num=1, eval_agent = agent):
    eval_env = np.zeros((env_size, env_size))
    eval_env = exploreEnv(eval_env, 15)

    scores = 0
    s = eval_env.reset_env(obstacle_num)
    done = eval_env.done

    while not done:
        frame = (s * 255).astype(np.uint8)
        video_writer.write(frame)

        action_probs = eval_agent.select_action(s)
        m = Categorical(action_probs)
        a = m.sample().item()
        s_prime, r, done = eval_env.explore(a)
        s = s_prime
    scores += r
    video_writer.release()

    if os.path.exists(video_file):
        print(f"비디오가 성공적으로 저장되었습니다: {video_file}")
    else:
        print("비디오 저장 실패")
    return scores


while total_steps < max_step:
    #print(total_steps)

    state = exploration.reset_env(obstacle_num)
    state = np.tile(state, (1, 1, 1))
    for i in range(horizon):
        action_probs = agent.select_action(state)
        m = Categorical(action_probs)
        a = m.sample().item()
        next_state, reward, done = exploration.explore(a)
        next_state = np.tile(next_state, (1, 1, 1))
        agent.put_data((state, a, reward, next_state, action_probs[0][a].item(), done))

        state = next_state
    
        if done:
            wandb.log({"Exploration Boundary " : exploration.final_observation})
            if total_steps % eval_interval == 0:
                rewards = evaluate()
                print("Steps: {}  AvgReturn: {}".format(total_steps, rewards))
                history['Step'].append(total_steps)
                history['AvgReturn'].append(rewards)

                clear_output()
                plt.figure(figsize=(8, 5))
                plt.plot(history['Step'], history['AvgReturn'], 'r-')
                plt.xlabel('Step', fontsize=16)
                plt.ylabel('AvgReturn', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(axis='y')
                plt.show(block=False)
                plt.pause(1)
                plt.close()

                if rewards > max_score:
                    max_score = rewards
                    torch.save(agent.actor.state_dict(), f'{train_name}.pt')
                wandb.log({"Evaluation Rewards" : rewards})
            break
    
    #print(f'Train steps : {total_steps} reward : {reward}')

    wandb.log({"Training Rewards" : reward})

    agent.update_model()
    total_steps += 1



test_agent = PPO(state_dim, action_dim)
test_agent.actor.load_state_dict(torch.load(f'{train_name}.pt'))    

rewards = evaluate(n_evals=3, eval_agent=test_agent)
print("Test Score:", rewards)

rewards = visualize(eval_agent=test_agent)
print("Visualize score:", rewards)