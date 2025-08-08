import numpy as np

from extensive_form_game.treeplex import TreeplexDomain


class EquilibriumAlgorithm:
    def __init__(self, game, name=None):
        self._game = game
        self._x = game.domain(0).center()
        self._y = game.domain(1).center()
        self._x_perturbed = self._x.copy()
        self._y_perturbed = self._y.copy()
        self._gradient_computations = 0

        self._name = name if name is not None else self.__class__.__name__
        if isinstance(game.domain(0), TreeplexDomain):
            self.is_efg = True
        else:
            self.is_efg = False
        self._epsilon = 0

        self._x_tremble = self._x.copy()
        self._y_tremble = self._y.copy()
        self._max_regret = self.max_info_set_regret()

    def profile(self):
        return self._x, self._y

    def exploitability(self, regularizer_x=0, regularizer_y=0):
        exp, _, _, _, _ = self._game.profile_exploitability(self._x, self._y, regularizer_x, regularizer_y)
        return exp

    def distance(self, x1, x2, player):
        if player == 0:
            seq1 = self._game.domain(0).sequence_form(x1)
            seq2 = self._game.domain(0).sequence_form(x2)
        else:
            seq1 = self._game.domain(1).sequence_form(x1)
            seq2 = self._game.domain(1).sequence_form(x2)
        return np.sqrt(np.sum(seq1 - seq2) ** 2)

    def profile_value(self):
        val = self._game.profile_value(self._x, self._y)
        if self.is_efg:
            return val[0]
        else:
            return val

    def iterate(self, num_iterations=1):
        raise NotImplementedError

    def gradient_computations(self):
        return self._gradient_computations

    def max_info_set_regret(self):
        if self.is_efg:
            return self._game.max_infoset_regret(self._x, self._y, self._epsilon)
        else:
            return 0

    def __repr__(self):
        return self._name
