import numpy as np
import random

class exploreEnv(object):
    def __init__(self, env, policy, prob, observe):
        self.env = env
        self.size = len(env)
        self.policy = policy
        self.prob = prob
        self.observe = observe
        self.state = [1, 1]
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

    def reset_env(self):
        self.env = np.zeros((self.size, self.size))
        self.state = [1, 1]

    # 장애물 설정
    def set_obstacle(self, obstacle, l):
        self.obstacle.append([obstacle[0], obstacle[1], l])
        
        for i in range(obstacle[0] - l, obstacle[0] + l + 1):
            for j in range(obstacle[1] - l, obstacle[1] + l + 1):
                if pow(i, 2) + pow(j, 2) <= pow(l, 2):
                    self.env[j, i] = 1
                    self.observable_area -= 1

    def explore(self, a):
        # action 선택 방법 구현 및 next_action에 할당. 현재는 임의 값으로 선언
        next_action = self.action[a]
        self.state[0] += next_action[0]
        self.state[1] += next_action[1]
        self.recent_action[0] = self.recent_action[1]
        self.recent_action[1] = a
        self.observation()


    # 이동한 위치에서의 reward 계산
    def observation(self):
        positive_reward=0
        negative_reward=0
        for i in range(max([0, self.state[0] - self.observe]), min([self.size, self.state[0] + self.observe] + 1)):
            for j in range(max([0, self.state[1] - self.observe]), min([self.size, self.state[1] + self.observe] + 1)):
                if pow(i, 2) + pow(j, 2) <= pow(self.observe, 2):
                    if self.env[j, i] == 0:
                        positive_reward += 1
                        self.env[j, i] = 0.5
                        self.observable_area -= 1
                    
                    elif self.env[j, i] == 1:
                        negative_reward -= 2
        
        if self.observable_area==0:
            positive_reward += 100
        elif positive_reward!=0:
            negative_reward -= self.observable_area * 5
                        
        return positive_reward + negative_reward
    
    # 여기 학습할 RL 모델 구현


env_size = 100
epoch = 20

env = np.zeros((env_size, env_size))

# policy, probablity는 사용할 RL 모델에 맞춰서 설정
exploration = exploreEnv(env, policy, prob, 8)



for i in epoch:
    exploration.reset_env()

    for i in range(4):
        obstacle = [random.randrange(20, 80), random.randrange(20, 80)]
        obstacle_size = random.randrange(5, 10)
        exploration.set_obstacle(obstacle, obstacle_size)


    