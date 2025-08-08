import importlib
import math
import os
import warnings

import numpy as np

from .eqm import EquilibriumAlgorithm

warnings.simplefilter("error", RuntimeWarning)


# type: 0
class MirrorDescent(EquilibriumAlgorithm):
    def __init__(self, game, prox_x=None, prox_y=None, is_last=False, rt_type=0, weight=1.0, rt_weight=None,
                 # rt have different form： \mu*(xr-x), mu*log(x/xr),
                 rt_step=1, epsilon=0, qpe=False, epsilon_qpe=0, name='Mirror Descent'):
        EquilibriumAlgorithm.__init__(self, game, name=name)

        self._prox_x = prox_x if prox_x is not None else game.domain(0).prox()
        self._prox_y = prox_y if prox_y is not None else game.domain(1).prox()

        self._is_last = is_last

        self._x = self._prox_x.center()
        self._y = self._prox_y.center()

        self._c_x = np.copy(self._x)
        self._c_y = np.copy(self._y)

        self._weight = weight

        self._t = 1
        self._init_rt_weight = rt_weight
        self._rt_weight = rt_weight
        self._rt_type = rt_type
        self._is_rt = True if rt_type != 0 else False
        self._r_x = self._x.copy()
        self._r_y = self._y.copy()
        self._rt_step = rt_step
        self._next_rt_num = 1
        self._update_distance = 0

        # for EFPE
        self._epsilon = epsilon
        self._prox_x.set_epsilon(self._epsilon)
        self._prox_y.set_epsilon(self._epsilon)

    def weight(self):
        return self._weight

    def rt_step(self):
        return self._rt_step

    def rt_weight(self):
        return self._init_rt_weight

    def update_distance(self):
        return self._update_distance

    def set_weights(self, weight):
        self._weight = weight

    @staticmethod
    def solve_log0(x):
        x[x < 10 ** (-16)] = 10 ** (-16)
        return x

    def iterate(self, num_iterations=2):
        for _ in range(num_iterations):
            self.take_step()

    def compute_rt(self, x, x_r, w):
        rt_methods = {0: lambda: np.zeros(len(x)), 1: lambda: (x_r - x) * w,  # Abe, Meng 2024 Mutant MWU, RT MWU
            2: lambda: np.log(self.solve_log0(x_r / self.solve_log0(x))) * w,  # Perolat 2021 R-NaD, OR-NaD
            3: lambda: (np.log(self.solve_log0(x)) + 1) * w,  # Liu 2023 Reg-OMWU
        }

        try:
            rt = rt_methods[self._rt_type]()
        except KeyError:
            raise ValueError(f"Invalid RT type {self._rt_type}")

        return rt

    def take_step(self):
        rt_x = self.compute_rt(self._c_x, self._r_x, self._rt_weight)  # compute rt term

        g_x = self._game.utility_for(0, self._c_y) + rt_x  # utility
        _, x = self._prox_x(-1, g_x, self._weight, self._c_x)  # min

        rt_y = self.compute_rt(self._c_y, self._r_y, self._rt_weight)  # compute rt term
        g_y = self._game.utility_for(1, self._c_x) + rt_y  # utility
        _, self._c_y = self._prox_y(-1, g_y, self._weight, self._c_y)  # min
        self._gradient_computations += 1

        self._c_x = x.copy()

        if self._is_last:
            self._x = self._c_x.copy()
            self._y = self._c_y.copy()
        else:
            self._x = self._game.domain(0).combine(self._x, 1 / self._t, self._c_x)
            self._y = self._game.domain(1).combine(self._y, 1 / self._t, self._c_y)
            self._gradient_computations += 1

        self._t += 1

        if self._is_rt and self._t == self._next_rt_num:
            self._next_rt_num += self._rt_step
            self._r_x = self._c_x.copy()
            self._r_y = self._c_y.copy()


# type: 1
class PredictiveMirrorDescent(MirrorDescent):
    def __init__(self, game, prox_x=None, prox_y=None, weight=1, is_last=False, rt_type=0, rt_weight=0, rt_step=2,
                 epsilon=0, name='PredictiveMirrorDescent'):
        MirrorDescent.__init__(self, game=game, prox_x=prox_x, prox_y=prox_y, is_last=is_last, weight=weight,
                               rt_type=rt_type, rt_weight=rt_weight, rt_step=rt_step, epsilon=epsilon)
        self._c_x_hat = self._c_x.copy()
        self._c_y_hat = self._c_y.copy()
        self._rt_weight = rt_weight
        self.exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                                regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))

        self._name = name

    def take_step(self):
        y = self._c_y
        g_x = self._game.utility_for(0, y,
                                     regularizer=self.compute_rt(self._c_x_hat, self._r_x, self._rt_weight))  # y^{t-1}
        _, _, self._c_x = self._prox_x(-1, g_x, self._weight, self._c_x_hat)

        x = self._c_x
        g_y = self._game.utility_for(1, x,
                                     regularizer=self.compute_rt(self._c_y_hat, self._r_y, self._rt_weight))  # x^{t-1}
        _, _, self._c_y = self._prox_y(-1, g_y, self._weight, self._c_y_hat)

        y = self._c_y
        g_x = self._game.utility_for(0, y,
                                     regularizer=self.compute_rt(self._c_x_hat, self._r_x, self._rt_weight))  # y^{t}
        _, _, self._c_x_hat = self._prox_x(-1, g_x, self._weight, self._c_x_hat)

        x = self._c_x
        g_y = self._game.utility_for(1, x,
                                     regularizer=self.compute_rt(self._c_y_hat, self._r_y, self._rt_weight))  # x^{t}
        _, _, self._c_y_hat = self._prox_y(-1, g_y, self._weight, self._c_y_hat)

        self._gradient_computations += 2

        if self._is_last:
            self._x = self._c_x.copy()
            self._y = self._c_y.copy()
        else:
            # uniform average
            self._x = self._game.domain(0).combine(self._x, 1 / self._t, self._c_x)
            self._y = self._game.domain(1).combine(self._y, 1 / self._t, self._c_y)
        self._t += 1

        # shrink tau:
        if self._rt_type == 3:
            exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                               regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))
            if exp < self.exp / 4:  # paper sets 4
                self.exp = exp
                self._rt_weight /= 2

        if self._is_rt and self._t == self._next_rt_num:
            self._next_rt_num += self._rt_step
            self._r_x = self._c_x.copy()
            self._r_y = self._c_y.copy()


# type：2
class DS_PredictiveMirrorDescent(MirrorDescent):

    def __init__(self, game, prox_x=None, prox_y=None, weight=1, is_last=False, rt_type=0, rt_weight=0, rt_step=2,
                 epsilon=0, name='DS_PredictiveMirrorDescent'):
        MirrorDescent.__init__(self, game=game, prox_x=prox_x, prox_y=prox_y, is_last=is_last, weight=weight,
                               rt_type=rt_type, rt_weight=rt_weight, rt_step=rt_step, epsilon=epsilon)
        # print(self._x)
        self._ds_br_x = game.domain(0).ds_br()
        self._ds_br_y = game.domain(1).ds_br()

        self._c_x_hat = self._c_x.copy()
        self._c_y_hat = self._c_y.copy()
        self._x_first = self._c_x.copy()  # store the 1-st strategy
        self._y_first = self._c_y.copy()
        # REG_CFR need multiply the information set weight

        self._w_seq_x = game.domain(0).seq_w()
        self._w_seq_y = game.domain(1).seq_w()

        self._rt_weight = rt_weight
        self._rt_weight_x = self._rt_weight * self._w_seq_x
        self._rt_weight_y = self._rt_weight * self._w_seq_y

        self._sum_delta_x = np.zeros(game.domain(0).num_information_sets())  # history delta for compute lambda
        self._sum_delta_y = np.zeros(game.domain(1).num_information_sets())

        #  lambda=\sqrt{\kai+sum_1^{t-1}{\delta^s}} default kai=1,
        self._lambda_1_x = np.ones(game.domain(0).num_information_sets())  # lambda t-1/2
        self._lambda_2_x = np.ones(game.domain(0).num_information_sets())  # lambda t+1/2

        self._lambda_1_y = np.ones(game.domain(1).num_information_sets())
        self._lambda_2_y = np.ones(game.domain(1).num_information_sets())

        # \delta^s=\|v_{s+1/2}-v_{s-1/2}\|_2^2
        # self._v_x = np.zeros(game.domain(0).dimension()) # todo , initial value default?
        self._v_x = self._game.utility_for(0, self._c_y)
        # self._v_2_x = np.zeros(game.domain(0).dimension())

        # self._v_y = np.zeros(game.domain(1).dimension())
        self._v_y = self._game.utility_for(1, self._c_x)
        # self._v_2_y = np.zeros(game.domain(1).dimension())

        self.exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                                regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))
        self._name = name

    def take_step(self):
        g_x = self._game.utility_for(0, self._c_y_hat,
                                     regularizer=self.compute_rt(self._c_x, self._r_x, self._rt_weight_x))  # y^{t-1}

        _, _, self._c_x = self._ds_br_x(-1, g_x, self._lambda_1_x, self._lambda_2_x, self._c_x,
                                        self._x_first)  # alpha, g, beta, gamma, y, z

        g_y = self._game.utility_for(1, self._c_x_hat,
                                     regularizer=self.compute_rt(self._c_y, self._r_y, self._rt_weight_y))  # x^{t-1}
        _, _, self._c_y = self._ds_br_y(-1, g_y, self._lambda_1_y, self._lambda_2_y, self._c_y, self._y_first)

        g_x = self._game.utility_for(0, self._c_y_hat,
                                     regularizer=self.compute_rt(self._c_x, self._r_x, self._rt_weight_x))  # y^{t}
        _, v_x, c_x_hat_copy = self._prox_x(-1, g_x, self._lambda_2_x, self._c_x)
        g_y = self._game.utility_for(1, self._c_x_hat,
                                     regularizer=self.compute_rt(self._c_y, self._r_y, self._rt_weight_y))  # x^{t}
        _, v_y, self._c_y_hat = self._prox_y(-1, g_y, self._lambda_2_y, self._c_y)

        self._c_y_hat = c_x_hat_copy

        delta_x = self._game.domain(0).update_v(v_x - self._v_x)
        self._v_x = v_x
        self._sum_delta_x += delta_x
        self._lambda_1_x = self._lambda_2_x.copy()
        self._lambda_2_x = np.sqrt(1 + self._sum_delta_x) * self._game.domain(0).info_w()

        delta_y = self._game.domain(1).update_v(v_y - self._v_y)
        self._v_y = v_y
        self._sum_delta_y += delta_y
        self._lambda_1_y = self._lambda_2_y.copy()
        self._lambda_2_y = np.sqrt(1 + self._sum_delta_y) * self._game.domain(1).info_w()

        self._gradient_computations += 2

        if self._is_last:
            self._x = self._c_x.copy()
            self._y = self._c_y.copy()
        else:
            # uniform average
            self._x = self._game.domain(0).combine(self._x, 1 / self._t, self._c_x)
            self._y = self._game.domain(1).combine(self._y, 1 / self._t, self._c_y)
        self._t += 1

        # shrink tau:
        if True:
            exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                               regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))
            if exp < self.exp / 4:  # paper sets 4
                self.exp = exp
                self._rt_weight /= 2


# type:3
class PredictiveMirrorDescent_EFPE(MirrorDescent):

    def __init__(self, game, prox_x=None, prox_y=None, weight=1, is_last=False, rt_type=0, rt_weight=0, rt_step=2,
                 epsilon=0, T=1, name='PredictiveMirrorDescent_EFPE'):
        MirrorDescent.__init__(self, game=game, prox_x=prox_x, prox_y=prox_y, is_last=is_last, weight=weight,
                               rt_type=rt_type, rt_weight=rt_weight, rt_step=rt_step, epsilon=epsilon)

        self._c_x_hat = self._c_x.copy()
        self._c_y_hat = self._c_y.copy()
        self._w_seq_x = game.domain(0).seq_w()
        self._w_seq_y = game.domain(1).seq_w()

        self._k = 0

        self._num_seq = np.logspace(np.log10(1), np.log10(T), num=100, endpoint=True, base=10.0, dtype=None, axis=0)
        self._epsilon_seq = np.logspace(np.log10(0.1), np.log10(0.001), num=100, endpoint=True, base=10.0, dtype=None,
                                        axis=0)
        self._rt_weight_x = self._rt_weight * self._w_seq_x
        self._rt_weight_y = self._rt_weight * self._w_seq_y
        self.exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                                regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))
        self._name = name

    def take_step(self):
        g_x = self._game.utility_for(0, self._c_y, regularizer=self.compute_rt(self._c_x_hat, self._r_x,
                                                                               self._rt_weight_x))  # y^{t-1}

        _, _, self._c_x = self._prox_x(-1, g_x, self._weight + self._rt_weight,
                                       self._c_x_hat)  # alpha, g, beta, gamma, y, z

        g_y = self._game.utility_for(1, self._c_x, regularizer=self.compute_rt(self._c_y_hat, self._r_y,
                                                                               self._rt_weight_y))  # x^{t-1}
        _, _, self._c_y = self._prox_y(-1, g_y, self._weight + self._rt_weight, self._c_y_hat)

        g_x = self._game.utility_for(0, self._c_y,
                                     regularizer=self.compute_rt(self._c_x_hat, self._r_x, self._rt_weight_x))  # y^{t}
        _, v_x, self._c_x_hat = self._prox_x(-1, g_x, self._weight + self._rt_weight, self._c_x_hat)
        g_y = self._game.utility_for(1, self._c_x,
                                     regularizer=self.compute_rt(self._c_y, self._r_y, self._rt_weight_y))  # x^{t}
        _, v_y, self._c_y_hat = self._prox_y(-1, g_y, self._weight + self._rt_weight, self._c_y_hat)

        self._gradient_computations += 2

        if self._is_last:
            self._x = self._c_x.copy()
            self._y = self._c_y.copy()
        else:
            # uniform average
            self._x = self._game.domain(0).combine(self._x, 1 / self._t, self._c_x)
            self._y = self._game.domain(1).combine(self._y, 1 / self._t, self._c_y)

        exp = self.epsilon(regularizer_x=self.compute_rt(self._x, self._r_x, self._rt_weight),
                           regularizer_y=self.compute_rt(self._y, self._r_y, self._rt_weight))
        if exp < self.exp * 0.25:  # paper sets 1/4
            self.exp = exp
            self._rt_weight /= 2
        # shrink tau:
        if self._epsilon != 0 and self._t >= self._num_seq[
            self._k + 1]:  # paper “Learning Extensive-Form Perfect Equilibria in Two-Player Zero-Sum Sequential Games” does not use this decrease method
            if True:
                self._k += 1

                self._epsilon = self._epsilon_seq[self._k]

                self._prox_x.set_epsilon(self._epsilon)
                self._prox_y.set_epsilon(self._epsilon)

        if self._is_rt and self._t >= self._next_rt_num:
            if False:
                self._k += 1
                self._rt_step = int(self._beta ** self._k)
                self._epsilon = (1 - 1e-4) ** self._k
                self._rt_weight = self._epsilon ** 2
                self._rt_weight_x = self._rt_weight * self._w_seq_x
                self._rt_weight_y = self._rt_weight * self._w_seq_y

            self._next_rt_num += self._rt_step

            self._r_x = self._c_x.copy()
            self._r_y = self._c_y.copy()
        self._t += 1

        # if self.self._tk


def mirror_descent_init(mirror_type=0, weight=None, rt_weight=1, rt_step=1, is_tuned=False, tuned_num=20,
        highest_multiplier=10, lowest_multiplier=0.001, rt_weight_tuned=None, rt_step_tuned=None, iterate_num=200,
        **kwargs):
    if rt_step_tuned is None:
        rt_step_tuned = {rt_step}
    if rt_weight_tuned is None:
        rt_weight_tuned = {rt_weight}

    def mirror_type_function(mirror_type, game, weight, rt_weight, rt_step, **kwargs):
        mirror_class = {0: lambda: MirrorDescent(game, weight=weight, rt_weight=rt_weight, rt_step=rt_step, **kwargs),
            1: lambda: PredictiveMirrorDescent(game, weight=weight, rt_weight=rt_weight, rt_step=rt_step, **kwargs),
            2: lambda: DS_PredictiveMirrorDescent(game, weight=weight, rt_weight=rt_weight, rt_step=rt_step, **kwargs),
            3: lambda: PredictiveMirrorDescent_EFPE(game, weight=weight, rt_weight=rt_weight, rt_step=rt_step,
                                                    **kwargs),

        }
        try:
            mr = mirror_class[mirror_type]()
        except KeyError:
            raise ValueError(f"Invalid mirror type {mirror_type}")

        return mr

    def init(game):
        nonlocal weight, rt_weight, rt_step
        if is_tuned:
            dir_path = f"./tuned_par/{game}"
            os.makedirs(dir_path, exist_ok=True)  # 创建目录（如果不存在）
            file_path = f"{dir_path}/{kwargs['name']}.txt"
            algs_list = []

            if weight is not None:
                weights_list = {weight}
            else:
                log_step = (math.log(highest_multiplier) - math.log(lowest_multiplier)) / (tuned_num - 1)
                weights_list = [lowest_multiplier * math.exp(i * log_step) for i in range(tuned_num)]

            for weight in weights_list:
                for rt_weight in rt_weight_tuned:
                    for rt_step in rt_step_tuned:
                        kwargs_copy = kwargs.copy()
                        kwargs_copy.pop('epsilon', None)
                        algs_list.append(mirror_type_function(mirror_type, game, weight, rt_weight, rt_step, epsilon=0,
                                                              **kwargs_copy))

            weight, rt_weight, rt_step, _ = tuned_params(algs_list, iterate_num=iterate_num, file_path=file_path)
        return mirror_type_function(mirror_type, game, weight, rt_weight, rt_step, **kwargs)

    return init


def get_class(class_name, ):
    module_name = class_name.split('.')
    module = ".".join(module_name[:-1])
    return getattr(importlib.import_module(module), module_name[-1])


def tuned_params(algs_list, iterate_num=200, file_path=""):
    best_par = []
    best_eps = np.inf

    with open(file_path, "a") as f:
        f.write(f'\ninterate num: {iterate_num}########################################################\n')

        for t in range(len(algs_list)):
            alg = algs_list[t]
            alg.iterate(iterate_num)
            eps = alg.epsilon()
            par = [alg.weight(), alg.rt_weight(), alg.rt_step()]

            if eps < best_eps:
                best_par = par
                best_eps = eps

            log_message = f'tuned params: t: {t}, par: {par}, eps: {eps}, best_par: {best_par}, best_eps: {best_eps}\n'
            print(log_message)
            f.write(log_message)

    return best_par[0], best_par[1], best_par[2], best_eps
