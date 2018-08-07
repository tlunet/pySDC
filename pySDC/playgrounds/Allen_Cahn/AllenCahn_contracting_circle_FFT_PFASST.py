import pySDC.helpers.plot_helper as plt_helper
import matplotlib.ticker as ticker
from mpi4py import MPI

import dill
import os
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_multigrid_MPI import allinclusive_multigrid_MPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.playgrounds.Allen_Cahn.AllenCahn_monitor import monitor


# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


def setup_parameters():
    """
    Helper routine to fill in all relevant parameters

    Note that this file will be used for all versions of SDC, containing more than necessary for each individual run

    Returns:
        description (dict)
        controller_params (dict)
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = None

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
    sweeper_params['QE'] = ['EE']
    sweeper_params['spread'] = False

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['L'] = 1.0
    problem_params['nvars'] = None
    problem_params['eps'] = 0.04
    problem_params['radius'] = 0.25

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = monitor
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn2d_imex  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = rhs_imex_mesh  # pass data type for f
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_fft2d

    return description, controller_params


def run_variant(variant=None):
    """
    Routine to run particular SDC variant

    Args:
        variant (str): string describing the variant

    Returns:

    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters()

    # add stuff based on variant
    if variant == 'PFASST1':
        description['level_params']['nsweeps'] = [1, 1]
        description['problem_params']['nvars'] = [(128, 128), (32, 32)]
    elif variant == 'PFASST2':
        description['level_params']['nsweeps'] = [2, 1]
        description['problem_params']['nvars'] = [(128, 128), (32, 32)]
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    out = 'Working on %s variant...' % variant
    print(out)

    # setup parameters "in time"
    t0 = 0.0
    Tend = 0.032

    # instantiate controller
    controller = allinclusive_multigrid_MPI(controller_params=controller_params, description=description, comm=MPI.COMM_WORLD)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # plt_helper.plt.imshow(uinit.values)
    # plt_helper.plt.show()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # plt_helper.plt.imshow(uend.values)
    # plt_helper.plt.show()

    # filter statistics by variant (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % \
          (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])

    fname = 'data/AC_reference_FFT_Tend{:.1e}'.format(Tend) + '.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    err = np.linalg.norm(uref - uend.values, np.inf)
    print('Error vs. reference solution: %6.4e' % err)
    print()

    return stats


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # Loop over variants, exact and inexact solves
    results = {}
    for variant in ['PFASST1', 'PFASST2']:

        results[(variant, 'exact')] = run_variant(variant=variant)

    # dump result
    fname = 'data/results_PFASST_variants_AllenCahn_1E-03'
    file = open(cwd + fname + '.pkl', 'wb')
    dill.dump(results, file)
    file.close()
    assert os.path.isfile(cwd + fname + '.pkl'), 'ERROR: dill did not create file'

    # visualize
    # show_results(fname, cwd=cwd)


if __name__ == "__main__":
    main()
