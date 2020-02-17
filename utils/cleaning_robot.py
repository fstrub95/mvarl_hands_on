from finite_env import FiniteEnv
import numpy as np


class RobotEnv(FiniteEnv):
    """
    Enviroment with 2 states and 3 actions
    Args:
        gamma (float): discount factor
        seed    (int): Random number generator seed
    """

    def __init__(self, gamma=0.5, seed=42):
        # Set seed
        self.RS = np.random.RandomState(seed)

        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        Ns = 2
        Na = 3
        
        _P = np.array([[[1, 0], [3/4, 1/4], [0, 0]], [[0,1],[1,0], [1,0]]])
        _R = np.array([[0,1,0], [0, -1, 0]])

        # Initialize base class
        states = np.arange(Ns).tolist()
        action_sets = [np.arange(Na).tolist()]*Ns
        super().__init__(states, action_sets, P, gamma)

    def reward_func(self, state, action, next_state):
        return _R[state, action]

    def reset(self, s=0):
        self.state = s
        return self.state

    def step(self, action):
        next_state = self.sample_transition(self.state, action)
        reward = self.reward_func(self.state, action, next_state)
        done = False
        info = {}
        self.state = next_state

        observation = next_state
        return observation, reward, done, info

    def sample_transition(self, s, a):
        prob = self.P[s,a,:]
        s_ = self.RS.choice(self.states, p = prob)
        return s_

    @property
    def P(self):
        return _P

    @property
    def R(self):
        R = np.zeros((self.Ns, self.Na, self.Ns))
        for s in range(self.Ns):
            for a in range(self.Na):
                for sn in range(self.Ns):
                    R[s,a,sn] = self.reward_func(s,a,sn)
        return R
