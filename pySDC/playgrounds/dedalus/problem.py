#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem class for dedalus
"""
from pySDC.core.Problem import ptype

import numpy as np
from scipy.linalg import blas
from collections import deque

import dedalus.public as d3

from dedalus.core.system import CoeffSystem
from dedalus.core.evaluator import Evaluator
from dedalus.tools.array import apply_sparse


def State(cls, fields):
    return [f.copy() for f in fields]


class DedalusProblem(ptype):

    dtype_u = State
    dtype_f = State

    # Dummy class to trick Dedalus
    class DedalusTimeStepper:
        steps = 1
        stages = 1
        def __init__(self, solver):
            self.solver = solver

    def __init__(self, problem:d3.IVP, nNodes, collUpdate=False):
        solver = problem.build_solver(self.DedalusTimeStepper)
        self.solver = solver

        # From new version
        self.subproblems = [sp for sp in solver.subproblems if sp.size]
        self.evaluator:Evaluator = solver.evaluator
        self.F_fields = solver.F

        self.M = nNodes

        c = lambda: CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.MX0, self.RHS = c(), c()
        self.LX = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[c() for _ in range(self.M)] for _ in range(2)])

        # Attributes
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.dt = None
        self.firstEval = True
        self.init = True

        # Instantiate M solver, needed only for collocation update
        if collUpdate:
            for sp in solver.subproblems:
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                else:
                    sp.LHS = sp.M_min @ sp.pre_right
                sp.M_solver = solver.matsolver(sp.LHS, solver)
        self.collUpdate = collUpdate

    def stateCopy(self):
        return [u.copy() for u in self.solver.state]


    def computeMX0(self, state, MX0):
        """
        Compute MX0 term used in RHS of both initStep and sweep methods

        Update the MX0 attribute of the timestepper object.
        """
        self.evaluator.require_coeff_space(state)
        # Compute and store MX0
        MX0.data.fill(0)
        for sp in self.subproblems:
            spX = sp.gather_inputs(state)
            apply_sparse(sp.M_min, spX, axis=0, out=MX0.get_subdata(sp))


    def updateLHS(self, dt, qI, init=False):
        """Update LHS and LHS solvers for each subproblem

        Parameters
        ----------
        dt : float
            Time-step for the updated LHS.
        qI : 2darray
            QDeltaI coefficients.
        init : bool, optional
            Wether or not initialize the LHS_solvers attribute for each
            subproblem. The default is False.
        """
        # Update only if different dt
        if self.dt == dt:
            return

        # Attribute references
        solver = self.solver

        # Update LHS and LHS solvers for each subproblems
        for sp in solver.subproblems:
            if self.init:
                # Eventually instanciate list of solver (ony first time step)
                sp.LHS_solvers = [None] * self.M
                self.init = False
            for i in range(self.M):
                if solver.store_expanded_matrices:
                    # sp.LHS.data[:] = sp.M_exp.data + k_Hii*sp.L_exp.data
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                    self.axpy(a=dt*qI[i, i], x=sp.L_exp.data, y=sp.LHS.data)
                else:
                    sp.LHS = (sp.M_min + dt*qI[i, i]*sp.L_min)  # CREATES TEMPORARY
                sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)


    def evalLX(self, LX):
        """
        Evaluate LX using the current state, and store it

        Parameters
        ----------
        LX : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.

        Returns
        -------
        None.

        """
        # Attribute references
        solver = self.solver

        self.evaluator.require_coeff_space(solver.state)
        # Evaluate matrix vector product and store
        for sp in solver.subproblems:
            spX = sp.gather_inputs(solver.state)
            apply_sparse(sp.L_min, spX, axis=0, out=LX.get_subdata(sp))


    def evalF(self, F, time, dt, wall_time):
        """
        Evaluate the F operator from the current solver state

        Note
        ----
        After evaluation, state fields are left in grid space

        Parameters
        ----------
        time : float
            Time of evaluation.
        F : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.
        dt : float
            Current time step.
        wall_time : float
            Current wall time.
        """

        solver = self.solver

        # Evaluate non linear term on current state
        t0 = solver.sim_time
        solver.sim_time = time
        if self.firstEval:
            solver.evaluator.evaluate_scheduled(
                wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
            self.firstEval = False
        else:
            solver.evaluator.evaluate_group('F')
        # Store F evaluation
        for sp in solver.subproblems:
            sp.gather_outputs(solver.F, out=F.get_subdata(sp))
        # Put back initial solver simulation time
        solver.sim_time = t0

    def solveAndStoreState(self, iNode):
        """
        Solve LHS * X = RHS using the LHS associated to a given node,
        and store X into the solver state.
        It uses the current RHS attribute of the object.

        Parameters
        ----------
        iNode : int
            Index of the nodes.
        """
        # Attribute references
        solver = self.solver
        RHS = self.RHS

        for field in solver.state:
            field.preset_layout('c')

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)
            spX = sp.LHS_solvers[iNode].solve(spRHS)  # CREATES TEMPORARY
            sp.scatter_inputs(spX, solver.state)
