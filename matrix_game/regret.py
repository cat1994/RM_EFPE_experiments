import numpy as np


class RegretMatching:
    def __init__(self, dimension, epsilon=0):
        self._dimension = dimension
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)
        self.epsilon = epsilon
        self.epsilon_seq = np.full(self._dimension, self.epsilon)

    def set_epsilon(self, epsilon=0):
        self.epsilon_seq = np.full(self._dimension, epsilon)

    def __call__(self, utility, rt, **kwargs):
        value = np.dot(self.strategy, utility)
        utility += rt
        r = utility - (value + np.dot(self.strategy, rt))

        # compute the regret in coordinate space
        p = 1 - sum(self.epsilon_seq)
        a = np.tile(self.epsilon_seq, (len(self.epsilon_seq), 1))
        r = p * r + a @ r

        self.regret += r
        #np.maximum(self.regret, 0, out=self.regret)
        np.maximum(self.regret, 0, out=self.strategy)

        Z = np.sum(self.strategy)
        if Z <= 0.0:
            self.strategy.fill(1.0)
            Z = self._dimension

        self.strategy /= Z

        # EFPE: to normal strategy space
        # x=B@x
        for i in range(self._dimension):
            a[i][i] += p
        a_t = a.T
        self.strategy = a_t @ self.strategy
        return value

    def __str__(self):
        return 'RegretMatching'


class RegretMatchingPlus:
    def __init__(self, dimension, epsilon=0):
        self._dimension = dimension
        self.strategy = np.ones(dimension) / dimension

        self.epsilon = epsilon
        self.epsilon_seq = np.full(self._dimension, self.epsilon)
        self.regret = np.zeros(dimension)

    def set_epsilon(self, epsilon=0):
        self.epsilon_seq = np.full(self._dimension, epsilon)

    def __call__(self, utility, rt, **kwargs):
        value = np.dot(self.strategy, utility)
        utility += rt
        r = utility - (value + np.dot(self.strategy, rt))

        # compute the regret in coordinate space
        p = 1 - sum(self.epsilon_seq)
        a = np.tile(self.epsilon_seq, (len(self.epsilon_seq), 1))
        r = p * r + a @ r

        self.regret += r
        np.maximum(self.regret, 0, out=self.regret)
        np.maximum(self.regret, 0, out=self.strategy)

        Z = np.sum(self.strategy)
        if Z <= 0.0:
            self.strategy.fill(1.0)
            Z = self._dimension

        self.strategy /= Z

        # EFPE: to normal strategy space
        # x=B@x
        for i in range(self._dimension):
            a[i][i] += p
        a_t = a.T
        self.strategy = a_t @ self.strategy
        return value

    def __str__(self):
        return 'RegretMatching+'


def regret_matching_initializer():
    def init(domain, epsilon=0):
        return RegretMatching(domain.dimension(), epsilon=epsilon)

    return init


def regret_matching_plus_initializer():
    def init(domain, epsilon=0):
        return RegretMatchingPlus(domain.dimension(), epsilon=epsilon)

    return init
