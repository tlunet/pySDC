from __future__ import division
import numpy as np
from scipy.fftpack import fft, ifft

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class advectiondiffusion1d_imex(ptype):
    """
    Example implementing the unforced 1D advection diffusion equation with periodic BC in [-L/2, L/2] in spectral space

    Attributes:
        xvalues: grid points in space
        ddx: spectral operator for gradient
        lap: spectral operator for Laplacian
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'c', 'freq', 'nu', 'L']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advectiondiffusion1d_imex, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                                        params=problem_params)

        self.xvalues = np.array([i * self.params.L / self.params.nvars - self.params.L / 2.0
                                 for i in range(self.params.nvars)])

        kx = np.zeros(self.init)
        for i in range(0, int(self.init / 2) + 1):
            kx[i] = 2 * np.pi / self.params.L * i
        for i in range(int(self.init / 2) + 1, self.init):
            kx[i] = 2 * np.pi / self.params.L * (-self.init + i)
        self.ddx = kx * 1j
        self.lap = -kx ** 2

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        tmp_impl = fft(u.values)
        tmp_expl = tmp_impl.copy()

        tmp_impl *= self.params.nu * self.lap
        tmp_expl *= -self.params.c * self.ddx

        f.impl.values = np.real(ifft(tmp_impl))
        f.expl.values = np.real(ifft(tmp_expl))
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I+factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        tmp = fft(rhs.values)
        tmp /= (1.0 - self.params.nu * factor * self.lap)
        me.values = np.real(ifft(tmp))
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        if self.params.freq >= 0:
            omega = 2.0 * np.pi * self.params.freq
            me.values = np.sin(omega * (self.xvalues - self.params.c * t)) * np.exp(-t * self.params.nu * omega ** 2)
        else:
            np.random.seed(1)
            me.values = np.random.rand(self.params.nvars)
        return me
