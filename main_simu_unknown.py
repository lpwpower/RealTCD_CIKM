# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import argparse
import copy
import numpy as np

from realtcd.main import main

gpu_num = [5]
i_dataset = 10
coeff_interv_sparsity = 2e-3 # need to be increased 5e-8
lr = 1e-3
reg_coeff = 0.2 #0.8 # 0.065 # need to be increased 0.025
interv_knl = "unknown" # unknown
interv_type = "perfect"

data_name = '5_p1_smp6000' # 10_p1_smp5500, 5_p1_smp6000, '5_p1_smp3000'
data_path = f'./data_simu/{interv_type}/{data_name}/' # 5_p1_smp6000
llm_res_matrix = '5_p1_smp6000_res_matrix_data1_full' #'10_p1_smp5500_res_matrix_data1_full+' #'5_p1_smp6000_res_matrix_data1_full' #f'{data_name}_res_matrix'# "downstream_gpt4_basep_nodata_res_matrix" # "downstream_gpt4da_basep_res_matrix"
exp_path = f'./exp_simu/llm_{interv_knl}_{data_name}/{llm_res_matrix}/dataset{i_dataset}/interv{coeff_interv_sparsity}reg{reg_coeff}/' 

if data_name=='5_p1_smp6000': #'5_p1_smp3000'
    num_vars=10
    num_nodes=5
if data_name=='10_p1_smp5500':   
    num_vars = 20 # 10
    num_nodes = 10 # 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp-path', type=str, default=exp_path,
                        help='Path to experiments')
    parser.add_argument('--train', type=bool, default=True, # action="store_true",
                        help='Run `train` function, get /train folder')
    parser.add_argument('--retrain', action="store_true",
                        help='after to-dag or pruning, retrain model from scratch before reporting nll-val')
    parser.add_argument('--dag-for-retrain', default=None, type=str, help='path to a DAG in .npy \
                        format which will be used for retrainig. e.g.  /code/stuff/DAG.npy')
    parser.add_argument('--random-seed', type=int, default=625, help="Random seed for pytorch and numpy")

    # data
    parser.add_argument('--llm-res-matrix-path', type=str, default=f'./llm_res/{llm_res_matrix}.npy' , # default=None
                        help='Path to llm results')
    parser.add_argument('--data-path', type=str, default=data_path , # default=None
                        help='Path to data files')
    parser.add_argument('--i-dataset', type=str, default=i_dataset, # default=None
                        help='dataset index')
    parser.add_argument('--num-vars', type=int, default=num_vars, # required=True, 
                        help='Number of variables')
    parser.add_argument('--num-nodes', type=int, default=num_nodes, # required=True, 
                        help='Number of nodes') # lpw add
    parser.add_argument('--train-samples', type=int, default=0.8,
                        help='Number of samples used for training (default is 80% of the total size)')
    parser.add_argument('--test-samples', type=int, default=None,
                        help='Number of samples used for testing (default is whatever is not used for training)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='number of folds for cross-validation')
    parser.add_argument('--fold', type=int, default=0,
                        help='fold we should use for testing')
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help='number of samples in a minibatch')
    parser.add_argument('--num-train-iter', type=int, default=1000000,
                        help='number of meta gradient steps')
    parser.add_argument('--normalize-data', type=bool, default=True, # action="store_true",
                        help='(x - mu) / std')
    parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                        help='When loading data, will remove some regimes from data set')
    parser.add_argument('--test-on-new-regimes', action="store_true",
                        help='When using --regimes-to-ignore, we evaluate performance on new regimes never seen during'
                             ' training (use after retraining).')

    # model
    parser.add_argument('--model', type=str, default='DCDI-G', # required=True,
                        help='model class (DCDI-G or DCDI-DSF)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument('--hid-dim', type=int, default=16,
                        help="number of hidden units per layer")
    parser.add_argument('--nonlin', type=str, default='leaky-relu',
                        help="leaky-relu | sigmoid")
    parser.add_argument("--flow-num-layers", type=int, default=2,
                        help='number of hidden layers of the DSF')
    parser.add_argument("--flow-hid-dim", type=int, default=16,
                        help='number of hidden units of the DSF')


    # intervention
    parser.add_argument('--intervention', default=True, # action="store_true",
                        help="Use data with intervention")
    parser.add_argument('--dcd', action="store_true",
                        help="Use DCD (DCDI with a loss not taking into account the intervention)")
    parser.add_argument('--intervention-type', type=str, default=interv_type,
                        help="Type of intervention: perfect or imperfect")
    parser.add_argument('--intervention-knowledge', type=str, default=interv_knl,
                        help="If the targets of the intervention are known or unknown")
    parser.add_argument('--coeff-interv-sparsity', type=float, default=coeff_interv_sparsity, #1e-8
                        help="Coefficient of the regularisation in the unknown \
                        interventions case (lambda_R)")

    # optimization
    parser.add_argument('--optimizer', type=str, default="rmsprop",
                        help='sgd|rmsprop')
    parser.add_argument('--lr', type=float, default=lr, # default=1e-3
                        help='learning rate for optim')
    parser.add_argument('--lr-reinit', type=float, default=None,
                        help='Learning rate for optim after first subproblem. Default mode reuses --lr.')
    parser.add_argument('--lr-schedule', type=str, default=None,
                        help='Learning rate for optim, change initial lr as a function of mu: None|sqrt-mu|log-mu')
    parser.add_argument('--stop-crit-win', type=int, default=100,
                        help='window size to compute stopping criterion')
    parser.add_argument('--reg-coeff', type=float, default=reg_coeff,
                        help='regularization coefficient (lambda)')

    # Augmented Lagrangian options
    parser.add_argument('--omega-gamma', type=float, default=1e-4,
                        help='Precision to declare convergence of subproblems')
    parser.add_argument('--omega-mu', type=float, default=0.9,
                        help='After subproblem solved, h should have reduced by this ratio')
    parser.add_argument('--mu-init', type=float, default=1e-8,
                        help='initial value of mu')
    parser.add_argument('--mu-mult-factor', type=float, default=2,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument('--gamma-init', type=float, default=0.,
                        help='initial value of gamma')
    parser.add_argument('--h-threshold', type=float, default=1e-8,
                        help='Stop when |h|<X. Zero means stop AL procedure only when h==0')

    # misc
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience in --retrain.')
    parser.add_argument('--train-patience', type=int, default=5,
                        help='Early stopping patience in --train after constraint')
    parser.add_argument('--train-patience-post', type=int, default=5,
                        help='Early stopping patience in --train after threshold')

    # logging
    parser.add_argument('--plot-freq', type=int, default=5000,
                        help='plotting frequency')
    parser.add_argument('--no-w-adjs-log', action="store_true",
                        help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')
    parser.add_argument('--plot-density', action="store_true",
                        help='Plot density (only implemented for 2 vars)')

    # device and numerical precision
    parser.add_argument('--gpu', type=bool, default=True, # action="store_true",
                        help="Use GPU")
    parser.add_argument('--gpu-num', type=int, default=gpu_num, nargs='+', 
                        help="used gpu")
    parser.add_argument('--float', action="store_true",
                        help="Use Float precision")

    main(parser.parse_args())
    print("coeff_interv_sparsity:",coeff_interv_sparsity)
