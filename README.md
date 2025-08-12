# Prerequisites
- Python 3.7 or higher

# Overview
This repository contains experimental code for the paper *"Last-Iterate Convergence in Adaptive Regret Minimization for Approximate Extensive-Form Perfect Equilibrium"* (ECAI-25).

# Running the Code
Below are example commands for running the experiments.

### 1. Compute a solution using CFR
`python driver.py -a cfr -t 500 --num_output 51 -g kuhn -r 3`

### 2. Compute a solution using EGT
`python driver.py -a egt_1e-3 -t 500 -w kroer17 --num_output 51 --prox_scalar 1.0 --init_gap 0.001 --allowed_exp_increase 1.1 -g kuhn -r 3`

### 3. Compute solutions using a sequence  of algorithms
`python driver.py -a egt_0,egt_1e-3,cfr+_0,cfr+_1e-3,rtcfr+_0,rtcfr+_1e-1,rtcfr+_1e-2,rtcfr+_1e-3,rtcfr+_adp,regomwu_0,regomwu_1e-3,regomwu_efpe_adp -t 500 -w kroer17 --num_output 51 --prox_scalar 1.0 --init_gap 0.001 --allowed_exp_increase 1.1 -g kuhn -r 3`
