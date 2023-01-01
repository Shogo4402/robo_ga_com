import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetricAnimation
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights

## アルゴリズムの性能評価
def evaluation_algorithm(res,hv,rm,ref_point):
    X, F = res.opt.get("X", "F")
    hist = res.history
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
        # retrieve the optimum from the algorithm
        opt = algo.opt
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    data_hv = None
    if hv:
        data_hv = print_HV(approx_ideal,approx_nadir,hist_F,n_evals,ref_point)
    if rm:
        print_RM(res)
    return data_hv
        

    
def print_HV(approx_ideal,approx_nadir,hist_F,n_evals,ref_point):
    ## HyperVolume
    metric = Hypervolume(ref_point=ref_point,norm_ref_point=False,zero_to_one=True,
                     ideal=approx_ideal,nadir=approx_nadir)
    hv = [metric.do(_F) for _F in hist_F]
    plt.figure(figsize=(3, 3))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()
    return [n_evals,hv,ref_point]

def print_RM(res):
    ## RunningMetric
    running = RunningMetricAnimation(delta_gen=5,n_plots=10,key_press=False,do_show=True)
    for algorithm in res.history:
        running.update(algorithm)

def print_object_space(res,problem,dim1,dim2):
    F = res.F
    xl, xu = problem.bounds()
    plt.figure(figsize=(4, 3))
    plt.scatter(F[:, dim1], F[:, dim2], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()

def print_norm_object_space(res,dim1,dim2):
    F = res.F
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    F = (F - approx_ideal) / (approx_nadir - approx_ideal)
    fl = F.min(axis=0)
    fu = F.max(axis=0)
    print(f"Scale f1: [{fl[dim1]}, {fu[dim1]}]")
    print(f"Scale f2: [{fl[dim2]}, {fu[dim2]}]")

    plt.figure(figsize=(4, 3))
    plt.scatter(F[:, dim1], F[:, dim2], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()

def print_asf(res,weights,dim1,dim2):
    F = res.F
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    F = (F - approx_ideal) / (approx_nadir - approx_ideal)
    decomp = ASF()
    i = decomp.do(F, 1/weights).argmin()
    print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))
    plt.figure(figsize=(4, 3))
    plt.scatter(F[:, dim1], F[:, dim2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[i, dim1], F[i, dim2], marker="x", color="red", s=200)
    plt.title("Objective Space")
    plt.show()

def print_pseudo_weights(res,weights,dim1,dim2):
    F = res.F
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    F = (F - approx_ideal) / (approx_nadir - approx_ideal)
    i = PseudoWeights(weights).do(F)
    print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))
    plt.figure(figsize=(4, 3))
    plt.scatter(F[:, dim1], F[:, dim2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[i, dim1], F[i, dim2], marker="x", color="red", s=200)
    plt.title("Objective Space")
    plt.show()