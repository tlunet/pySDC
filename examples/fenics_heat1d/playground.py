import dolfin as df
import numpy as np

from pySDC.controller_classes.PFASST_blockwise_serial import PFASST_blockwise_serial
from examples.fenics_heat1d.ProblemClass import fenics_heat
from examples.fenics_heat1d.TransferClass import mesh_to_mesh_fenics
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats
from pySDC.datatype_classes.fenics_mesh import fenics_mesh,rhs_fenics_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 5E-09

    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    # pparams['c_nvars'] = [(16,16)]
    pparams['c_nvars'] = [128]
    pparams['family'] = 'CG'
    pparams['order'] = [1]
    pparams['refinements'] = [1,0]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = 3

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_heat
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_fenics
    description['transfer_params'] = tparams

    # initialize controller
    PFASST = PFASST_blockwise_serial(num_procs=num_procs, step_params=sparams, description=description)

    # setup parameters "in time"
    t0 = PFASST.MS[0].levels[0].prob.t0
    dt = 0.5
    Tend = 1*dt

    # get initial values on finest level
    P = PFASST.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = PFASST.run(u0=uinit, t0=t0, dt=dt, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('(classical) error at time %s: %s' %(Tend,abs(uex-uend)/abs(uex)))


    uex = df.Expression('sin(a*x[0]) * cos(t)',a=np.pi,t=Tend)
    print('(fenics-style) error at time %s: %s' %(Tend,df.errornorm(uex,uend.values)))

    extract_stats = grep_stats(stats,iter=-1,type='residual')
    sortedlist_stats = sort_stats(extract_stats,sortby='step')
    print(extract_stats,sortedlist_stats)