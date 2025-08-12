from __future__ import print_function

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from eqm import excessive_gap_technique as egt
from eqm import mirror_descent as md
from eqm import regret as eqm_regret
from games import goofspiel, kuhn, leduc
from games import liars_dice
from games import matrix_game
from matrix_game import regret as matrix_regret


def save_arrays_to_file(array1, array2, file_path):
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = {'array1': array1.tolist(), 'array2': array2.tolist()}
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


algs = {'EGT': lambda args: egt.excessive_gap_technique_init(aggressive_stepsizes=args.aggressive_stepsizes,
                                                             init_gap=init_gap, init_update_x=init_update_x,
                                                             allowed_exp_increase=allowed_exp_increase, ),
        'EGT_0': lambda args: egt.excessive_gap_technique_init(aggressive_stepsizes=args.aggressive_stepsizes,
                                                               init_gap=init_gap, init_update_x=init_update_x,
                                                               allowed_exp_increase=allowed_exp_increase,
                                                               epsilon=1e-15),
        'EGT_1E-3': lambda args: egt.excessive_gap_technique_init(aggressive_stepsizes=args.aggressive_stepsizes,
                                                                  init_gap=init_gap, init_update_x=init_update_x,
                                                                  allowed_exp_increase=allowed_exp_increase,
                                                                  epsilon=1e-3),

        # vanilla CFR, CFR+
        'CFR': lambda args: eqm_regret.regret_minimization_initializer(matrix_regret.regret_matching_initializer(),
                                                                       alternate=True, averaging='quadratic',
                                                                       name='CFR'),
        'CFR+_0': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), alternate=True, averaging='quadratic', name='CFR+',
            epsilon=1e-15), 'CFR+_1E-3': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.regret_matching_plus_initializer(), alternate=True, averaging='quadratic', name='CFR+',
        epsilon=1e-3),

        # RT-CFR
        'RTCFR': lambda args: eqm_regret.regret_minimization_initializer(matrix_regret.regret_matching_initializer(),
                                                                         alternate=True, is_last=True, rt_type=1,
                                                                         rt_step=10, rt_weight=0.001, name='RTCFR'),

        'RTCFR+_0': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), alternate=True, is_last=True, rt_type=1, rt_step=10,
            rt_weight=0.001, is_tuned=False, rt_weight_tuned=[0.005, 0.001], rt_step_tuned=[20, 30, 40],
            iterate_num=100, epsilon=0, name='RTCFR+'),

        'RTCFR+_1E-1': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), alternate=True, is_last=True, rt_type=1, rt_step=10,
            rt_weight=0.001, is_tuned=False, rt_weight_tuned=[0.005, 0.001], rt_step_tuned=[20, 30, 40],
            iterate_num=100, epsilon=1e-1, name='RTCFR+'),

        'RTCFR+_1E-2': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), alternate=True, is_last=True, rt_type=1, rt_step=10,
            rt_weight=0.001, is_tuned=False, rt_weight_tuned=[0.005, 0.001], rt_step_tuned=[20, 30, 40],
            iterate_num=100, epsilon=1e-2, name='RTCFR+'),

        'RTCFR+_1E-3': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), alternate=True, is_last=True, rt_type=1, rt_step=10,
            rt_weight=0.001, is_tuned=False, rt_weight_tuned=[0.005, 0.001], rt_step_tuned=[20, 30, 40],
            iterate_num=100, epsilon=1e-3, name='RTCFR+'),

        'RTCFR+_ADP': lambda args: eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(), adaptive_epsilon=True, alternate=True, is_last=True,
            rt_type=1, rt_step=5, rt_weight=0.01, is_tuned=False, rt_weight_tuned=[0.005, 0.001],
            rt_step_tuned=[20, 30, 40], iterate_num=100, epsilon=1e-1, delta=1, gamma=0.5, name='RTCFR+'),

        'MWU': lambda args: md.mirror_descent_init(weight=None, is_tuned=True, tuned_num=20, highest_multiplier=10,
                                                   lowest_multiplier=0.01, iterate_num=100, name='MWU'),
        'OMWU_0': lambda args: md.mirror_descent_init(mirror_type=1, weight=None, is_last=True, is_tuned=True,
                                                      tuned_num=20, highest_multiplier=10, lowest_multiplier=0.01,
                                                      iterate_num=100, epsilon=1e-15, name='OMWU'),

        'REGOMWU': lambda args: md.mirror_descent_init(mirror_type=1, weight=None, is_last=True, rt_type=3,
                                                       rt_weight=1e-5, is_tuned=True, tuned_num=20,
                                                       highest_multiplier=1, lowest_multiplier=1e-5,
                                                       rt_weight_tuned=[1e-5], iterate_num=100, name='Reg-OMWU'),
        'REGOMWU_0': lambda args: md.mirror_descent_init(mirror_type=1, weight=None, is_last=True, rt_type=3,
                                                         rt_weight=1e-5, is_tuned=True, tuned_num=20,
                                                         highest_multiplier=1, lowest_multiplier=1e-5,
                                                         rt_weight_tuned=[1e-5], iterate_num=100, epsilon=1e-15,
                                                         name='Reg-OMWU'),
        'REGOMWU_1E-3': lambda args: md.mirror_descent_init(mirror_type=1, weight=None, is_last=True, rt_type=3,
                                                            rt_weight=1e-5, is_tuned=True, tuned_num=20,
                                                            highest_multiplier=1, lowest_multiplier=1e-5,
                                                            rt_weight_tuned=[1e-5], iterate_num=100, epsilon=1e-3,
                                                            name='Reg-OMWU'),
        'REGOMWU_EFPE_ADP': lambda args: md.mirror_descent_init(mirror_type=3, weight=None, is_last=True, rt_type=3,
                                                                rt_weight=1e-5, is_tuned=True, tuned_num=20,
                                                                highest_multiplier=1, lowest_multiplier=1e-5,
                                                                rt_weight_tuned=[1e-5], iterate_num=100, epsilon=0.1,
                                                                T=500, name='REGOMWU_EFPE'), }

parser = argparse.ArgumentParser()
# Game params
parser.add_argument('-g', '--game', default='kuhn', dest='game',
                    help='Game to solve: kuhn, leduc, goofspile, liars_dice')
parser.add_argument('-r', '--num_ranks', type=int, default=3, help='Number of ranks in the deck.')

parser.add_argument('--seed', type=int, default=1, help='Random seed for matrix game.')
parser.add_argument('--dimension', nargs='+', type=int, help='matrix dimension')

# Algorithm params
parser.add_argument('-a', '--algorithm', default=','.join(algs.keys()), dest='alg',
                    help='Available algorithms: %s' % ', '.join(
                        algs.keys()) + '. (a comma-separated list chooses several algorithms, ' + 'e.g. \'EGT,CFR+\')')
parser.add_argument('-t', '--num_iterations', type=int, default=100, help='number of algorithm iterations')
parser.add_argument('--exp_threshold', type=float, default=-1.0, help='stopping threshold')
parser.add_argument('-s', '--aggressive_stepsizes', action='store_true', default=False, dest='aggressive_stepsizes',
                    help='use aggressive stepsizing in EGT')

# DGF params
parser.add_argument('--prox_scalar', type=float, default=1, help='Scalar value applied to the whole prox function.\
                    Default uses 1.0')
parser.add_argument('-w', '--prox_infoset_weights', default="all_one", dest='prox_infoset_weights', help='The weighting scheme used to construct entropy DGF. \
                    Option: all_one, kroer15, kroer17, farina21.')

# EGT-only params
parser.add_argument('--init_gap', type=float, default=-1.0,
                    help='find initial smoothing parameters that satisfy this gap')
parser.add_argument('--allowed_exp_increase', type=float, default=-1.0, help='If > 0 then disallow iterations that cause a decrease in solution\
            quality by more than the given multiplicative factor')
parser.add_argument('--init_update_x', action='store_true', default=False, dest='init_update_x',
                    help='whether to update initial x during search for initial' + 'smoothing parameters.')

# Args to do with output formatting
parser.add_argument('--num_outputs', type=int, default=10, help='number of outputs')
parser.add_argument('-d', '--debug', action='store_true', default=False, dest='debug', help='display debug output')
parser.add_argument('--csv', action='store_true', default=False, dest='to_csv', help='whether to output in CSV format')
parser.add_argument('--pretty_print', action='store_true', default=True, dest='pretty_print',
                    help='whether to output in CSV format')
parser.add_argument('--log_scale', action='store_true', default=False, dest='log_scale',
                    help='whether to output the num_outputs outputs spaced ' + 'according to linear or log-scale x axes.')

args = parser.parse_args()

num_iterations = args.num_iterations
num_outputs = args.num_outputs
exp_threshold = args.exp_threshold
debug = args.debug
to_csv = args.to_csv
pretty_print = args.pretty_print
log_scale = args.log_scale

file_path = "./plot_EFPE_test/%s/%s_%s_%s.txt" % (args.game, args.game, args.num_ranks, args.alg)

directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = file_path.replace('+', '_plus')
gnuplot_out = open(file_path, 'wt')

init_gap = args.init_gap
init_update_x = args.init_update_x
allowed_exp_increase = args.allowed_exp_increase
if to_csv:
    print('iters,gradients,exp,profile_val,algorithm,time')
elif debug:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

if args.game == 'kuhn_matrix':
    game = kuhn.init_matrix()
elif args.game == 'kuhn':
    game = kuhn.init_efg(num_ranks=args.num_ranks, prox_infoset_weights=args.prox_infoset_weights,
                         prox_scalar=args.prox_scalar)
elif args.game == 'leduc':
    game = leduc.init_efg(num_ranks=args.num_ranks, prox_infoset_weights=args.prox_infoset_weights,
                          prox_scalar=args.prox_scalar)
elif args.game == 'goofspiel':
    game = goofspiel.init_efg(num_ranks=args.num_ranks, prox_infoset_weights=args.prox_infoset_weights,
                              prox_scalar=args.prox_scalar)
elif args.game == 'liars_dice':
    game = liars_dice.init_efg(num_ranks=args.num_ranks, prox_infoset_weights=args.prox_infoset_weights,
                               prox_scalar=args.prox_scalar)
elif args.game == 'matrix_game':
    game = matrix_game.init_nfg(seed=args.seed, dimension=tuple(args.dimension))

else:
    assert False, 'unknown game %s' % args.game

algs_to_run = []

algs_arg = set(args.alg.upper().split(','))
for alg in algs_arg:
    if alg not in algs:
        print('Unknown algorithm "%s"' % alg)
        sys.exit(1)
    else:
        algs_to_run += [algs[alg](args)]

alg_names = []

if log_scale:
    print_seq = np.unique(np.insert(np.geomspace(1, num_iterations, num_outputs, dtype=int), 0, 0))
else:
    print_seq = np.linspace(0, num_iterations, num_outputs, dtype=int)
for alg_idx, alg in enumerate(algs_to_run):
    t0 = time.time()
    opt = alg(game)
    total_time = time.time() - t0
    if not to_csv:
        print(opt)
        print('iters\tgrads\texp\tprofile_val\ttime\tmax_infoset_regret')
    exp_initial = opt.exploitability()
    profile_val_initial = opt.profile_value()

    alg_names.append(str(opt))
    gradient_computations = opt.gradient_computations()

    t = 0

    max_info_set_regret = opt.max_info_set_regret()

    for i in range(len(print_seq)):
        if i > 0:
            num_iterations = print_seq[i] - print_seq[i - 1]
        else:
            num_iterations = print_seq[0]

        t0 = time.time()
        opt.iterate(num_iterations)
        total_time += time.time() - t0
        exp = opt.exploitability()

        if i == len(print_seq) - 1:
            x, y = opt.profile()
            save_arrays_to_file(x, y, './profile/%s_%s_%s.json' % (args.game, args.num_ranks, args.alg))
        profile_val = opt.profile_value()
        max_info_set_regret = opt.max_info_set_regret()
        if to_csv:
            print('{iters},{gradients},{exp},{profile_val},{algorithm},{time}'.format(iters=print_seq[i],
                                                                                      gradients=opt.gradient_computations(),
                                                                                      exp=exp, profile_val=profile_val,
                                                                                      algorithm=opt, time=total_time))
        elif pretty_print:
            if False and str(opt) == 'ExcessiveGapTechnique':
                print('{iters}\t{grads}\t{exp:.6f}\t{profile_val:.6f}\t{egv:.6f}'.format(iters=print_seq[i],
                                                                                         grads=opt.gradient_computations(),
                                                                                         exp=exp,
                                                                                         profile_val=profile_val,
                                                                                         egv=opt.excessive_gap(), ))
            else:
                print('{iters}\t{grads}\t{exp}\t{profile_val}\t{time}\t{max_info_set_regret}'.format(iters=print_seq[i],
                                                                                                     grads=opt.gradient_computations(),
                                                                                                     exp=exp,
                                                                                                     profile_val=profile_val,
                                                                                                     time=total_time,
                                                                                                     max_info_set_regret=max_info_set_regret))
        else:
            print(print_seq[i], opt.gradient_computations(), exp, profile_val)
        print(print_seq[i], opt.gradient_computations(), exp, total_time, max_info_set_regret, file=gnuplot_out)

        if exp < exp_threshold:
            break

gnuplot_out.close()
