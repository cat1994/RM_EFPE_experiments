import math
import os

import numpy as np

from extensive_form_game.cfr import CounterfactualRegretMinimizer
from extensive_form_game.treeplex import TreeplexDomain
from .eqm import EquilibriumAlgorithm

"""
returns the sequence {alpha + beta*sqrt(t) + gamma*t}_{t=1}^\\inf
"""


def step_size_generator(alpha, beta, gamma, eta=0):
    t = 1
    while True:
        t += 1
        yield alpha + beta * math.sqrt(t - 1) + gamma * (t - 1) + eta * (t - 1) ** 2


def entropy(x, y):
    return (np.array([i * np.log(i) if i != 0 else 0 for i in x]) - np.array(
        [np.log(i) if i != 0 else np.log(1e-8) for i in y])) * x


class RegretMinimization(EquilibriumAlgorithm):
    def __init__(self, game, rm_x, rm_y=None, is_last=False, alternate=False, step_param=None, rt_type=0, rt_weight=1.0,
                 rt_step=0, epsilon=0, adaptive_epsilon=False, gamma=1.0, name=None):
        if step_param is None:
            step_param = {'alpha': 1, 'beta': 0, 'gamma': 0}

        def _init_rm(domain, rm):  # RM for NFG or CFR for EFG
            if isinstance(domain, TreeplexDomain):
                regret_matcher = CounterfactualRegretMinimizer(domain, rm, epsilon, name)
            else:
                regret_matcher = rm(domain)
            self._name = str(regret_matcher)
            return regret_matcher

        rm_y = rm_x if rm_y is None else rm_y
        self._rm_x = _init_rm(game.domain(0), rm_x)
        self._rm_y = _init_rm(game.domain(1), rm_y)
        EquilibriumAlgorithm.__init__(self, game, name=self._name)

        self.adaptive_epsilon = adaptive_epsilon

        # adaptive EFPE
        # set the epsilon for every seq
        if self.adaptive_epsilon:
            # self._info_set_reach_seq_x = self._game.get_reach_matrix(0)
            # self._info_set_reach_seq_y = self._game.get_reach_matrix(1)
            # self._info_set_reach_seq_x[self._info_set_reach_seq_x != 0] = epsilon
            # self._info_set_reach_seq_y[self._info_set_reach_seq_y != 0] = epsilon
            #
            # self._epsilon_x = np.max(self._info_set_reach_seq_y, axis=0)
            # self._epsilon_y = np.max(self._info_set_reach_seq_x, axis=0)

            self._gamma = gamma
            self._delta=self.max_info_set_regret()

            # self._rm_x.set_epsilon_by_sequence(self._epsilon_x)
            # self._rm_y.set_epsilon_by_sequence(self._epsilon_y)

        # elif epsilon != 0:
        #     self._rm_x.set_epsilon(epsilon, info_w=None)  # todo may occur in rm without that function
        #     self._rm_y.set_epsilon(epsilon, info_w=None)

        self._is_last = is_last
        self._alternate = alternate

        # for average strategy
        self._step_param = step_param
        self._step = step_size_generator(**step_param)
        self._t = 1  # iterate times
        self._alpha = next(self._step)
        self._weight = self._alpha

        # rt module
        self._rt_type = rt_type
        self._rt_step = rt_step  # given fixed initial parameter
        self._reference_step = self._rt_step
        self._rt_weight = rt_weight
        self._rw = 1
        self._exp = self.exploitability()  # 设定为初始策略的exp
        self._reference_strategy_x = self._rm_x.strategy.copy()  # initial reference x and y to x0 and y0
        self._reference_strategy_y = self._rm_y.strategy.copy()

        self._avg_x = self._x.copy()
        self._avg_y = self._y.copy()

        # other parameter
        self._interval = 1
        # self._interval_list = [10]

        self._init_rt_weight = rt_weight
        self._epsilon = epsilon
        self._n = 1

        # self._epsilon_x_scale = self._epsilon
        # self._epsilon_y_scale = self._epsilon

    def rt_weight(self):
        return self._init_rt_weight

    def rt_step(self):
        return self._rt_step

    def reference_profile(self):
        return self._reference_strategy_x, self._reference_strategy_y

    @staticmethod
    def solve_log0(x):
        x[x < 10 ** (-16)] = 10 ** (-16)
        return x

    def compute_rt(self, x, x_r, w):
        rt_methods = {0: lambda: np.zeros(len(x)), 1: lambda: (x_r - x) * w,  # Abe, Meng 2024: Mutant MWU, RT MWU
                      2: lambda: np.log(self.solve_log0(x_r / self.solve_log0(x))) * w,  # Perolat 2021 R-NaD, OR-NaD
                      3: lambda: (np.log(self.solve_log0(x)) + 1) * w,  # Liu 2023 Reg-OMWU
                      }
        try:
            rt = rt_methods[self._rt_type]()
        except KeyError:
            raise ValueError(f"Invalid RT type {self._rt_type}")

        return rt

    def iterate(self, num_iterations=1):
        for t in range(num_iterations):
            if False and self._exp < 1e-13 and self._delta < 1e-13:  # set the condition, omit the iterations leave
                self._gradient_computations += 1
                continue

            # update regret and strategy
            rt_x = self.compute_rt(self._rm_x.strategy, self._reference_strategy_x,
                                   self._rt_weight * self._rw)  # x is a behavior strategy not seq
            rt_y = self.compute_rt(self._rm_y.strategy, self._reference_strategy_y, self._rt_weight * self._rw)
            y = self._rm_y.strategy
            u_x = self._game.utility_for(0, y)
            if not self._alternate:
                u_y = self._game.utility_for(1, y) + rt_y

            self._rm_x(u_x, rt_x, t=self._t)
            self._gradient_computations += 1
            x = self._rm_x.strategy
            if self._alternate:
                u_y = self._game.utility_for(1, x)

            self._rm_y(u_y, t=self._t, rt=rt_y)

            if self._is_last:  # using last
                self._x = self._rm_x.strategy.copy()
                self._y = self._rm_y.strategy.copy()
            else:
                self._alpha = next(self._step)
                self._weight += self._alpha

                alpha = self._alpha / self._weight
                self._x = self._game.domain(0).combine(self._x, alpha, self._rm_x.strategy)
                self._y = self._game.domain(1).combine(self._y, alpha, self._rm_y.strategy)

                self._x_tremble = self._game.domain(0).combine(self._x_tremble, alpha, x)
                self._y_tremble = self._game.domain(1).combine(self._y_tremble, alpha, y)
                self._gradient_computations += 1

            if self._rt_type != 0 and self._t == self._reference_step:
                self._exp = min(self._exp, self.exploitability())
                # EFPE  decay epsilon adaptively
                if self.adaptive_epsilon:
                    r_max=self.max_info_set_regret()
                    if r_max<=self._delta:
                        self._epsilon*=self._gamma
                        self._delta*=self._gamma
                        self._rm_x.set_epsilon(self._epsilon)
                        self._rm_y.set_epsilon(self._epsilon)

                    # bound1 = 0
                    # self._regret_base_x = max(bound1, self._regret_base_x * self._gamma)
                    # self._regret_base_y = max(bound1, self._regret_base_y * self._gamma)

                    # 设置epsilon
                    # self._epsilon_x *= self._gamma
                    # self._epsilon_y *= self._gamma
                    # bound2 = 0
                    # if max(self._epsilon_x) > bound2:
                    #     self._epsilon_x[self._epsilon_x != 0] = np.maximum(self._epsilon_x[self._epsilon_x != 0],
                    #                                                        bound2)
                    #
                    #     self._epsilon_y[self._epsilon_y != 0] = np.maximum(self._epsilon_y[self._epsilon_y != 0],
                    #                                                        bound2)
                        # self._rm_x.set_epsilon_by_sequence(self._epsilon_x)
                        # self._rm_y.set_epsilon_by_sequence(self._epsilon_y)
                        # print(f"regret x max: {max(regret_set_x)}, \t"
                        #       f"regret y max: {max(regret_set_y)}, \t"
                        #       f"epsilon: {max(self._epsilon_x)}, \t"
                        #       f"regret_base: {self._regret_base_x} \t")

                # adaptive RTCFR JMLR:

                else:
                    self._reference_strategy_x = self._rm_x.strategy.copy()
                    self._reference_strategy_y = self._rm_y.strategy.copy()
                self._reference_step += self._rt_step

            self._interval += 1
            self._t += 1


def regret_minimization_initializer(rm_x, rm_y=None, averaging='uniform', is_tuned=False, rt_weight=1.0, rt_step=1,
                                    rt_step_tuned=None, rt_weight_tuned=None, iterate_num=200, **kwargs):
    def init(game):
        nonlocal rt_weight, rt_step

        if is_tuned:
            dir_path = f"./tuned_par/{game}"
            os.makedirs(dir_path, exist_ok=True)
            file_path = f"{dir_path}/{kwargs['name']}.txt"
            algs_list = []
            for rt_weight in rt_weight_tuned:
                for rt_step in rt_step_tuned:
                    algs_list.append(RegretMinimization(game, rm_x, rt_weight=rt_weight, rt_step=rt_step, **kwargs))

            rt_weight, rt_step, _ = tuned_params(algs_list, iterate_num=iterate_num, file_path=file_path)

        step_param = {'alpha': 1 if averaging == 'uniform' else 0, 'beta': 0,
                      'gamma': 1 if averaging == 'linear' else 0, 'eta': 1 if averaging == 'quadratic' else 0}

        return RegretMinimization(game, rm_x, rt_weight=rt_weight, rt_step=rt_step, step_param=step_param, **kwargs)

    return init


def tuned_params(algs_list, iterate_num=200, file_path=""):
    best_par = []
    best_eps = np.inf

    with open(file_path, "a") as f:
        f.write(f'\ninterate num: {iterate_num}########################################################\n')
        for t in range(len(algs_list)):
            alg = algs_list[t]
            alg.iterate(iterate_num)
            eps = alg.epsilon()
            par = [alg.rt_weight(), alg.rt_step()]

            if eps < best_eps:
                best_par = par
                best_eps = eps

            log_message = f'tuned params: t: {t}, par: {par}, eps: {eps}, best_par: {best_par}, best_eps: {best_eps}\n'
            print(log_message)
            f.write(log_message)

    return best_par[0], best_par[1], best_eps
