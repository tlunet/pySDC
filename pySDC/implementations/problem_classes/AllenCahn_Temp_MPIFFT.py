import numpy as np
from mpi4py_fft import PFFT

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray


class allencahn_temp_imex(Problem):
    r"""
    This class implements the :math:`N`-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [0, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = D \Delta u - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w \frac{u - T_M}{T_M}u (1 - u)

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`, with driving force :math:`d_w`, and :math:`N=2,3`. :math:`D` and
    :math:`T_M` are fixed parameters. Different initial conditions can be used, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{(x_i-0.5)^2 + (y_j-0.5)^2}}{\sqrt{2}\varepsilon}\right),

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is treated
    *semi-implicitly*, i.e., the nonlinear system is solved by Fast-Fourier Tranform (FFT) and the linear parts in the right-hand
    side will be treated explicitly using ``mpi4py-fft`` [1]_ to solve them.

    Attributes
    ----------
    nvars : List of int tuples, optional
        Number of unknowns in the problem, e.g. ``nvars=[(128, 128), (64, 64)]``.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    radius : float, optional
        Radius of the circles.
    spectral : bool, optional
        Indicates if spectral initial condition is used.
    TM : float, optional
        Problem parameter :math:`T_M`.
    D : float, optional
        Problem parameter :math:`D`.
    dw : float, optional
        Driving force.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str, optional
        Initialises type of initial state.
    comm : bool, optional
        Communicator.

    Attributes
    ----------
    fft : fft object
        Object for FFT.
    X : np.ogrid
        Grid coordinates in real space.
    K2 : np.ndarray
        Laplace operator in spectral space.
    dx : float
        Mesh width in x direction.
    dy : float
        Mesh width in y direction.

    References
    ----------
    .. [1] https://mpi4py-fft.readthedocs.io/en/latest/
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self,
        nvars=None,
        eps=0.04,
        radius=0.25,
        spectral=None,
        TM=1.0,
        D=10.0,
        dw=0.0,
        L=1.0,
        init_type='circle',
        comm=None,
    ):
        """Initialization routine"""

        if nvars is None:
            nvars = [(128, 128)]

        if not (isinstance(nvars, tuple) and len(nvars) > 1):
            raise ProblemError('Need at least two dimensions')

        # creating FFT structure
        ndim = len(nvars)
        axes = tuple(range(ndim))
        self.fft = PFFT(comm, list(nvars), axes=axes, dtype=np.float64, collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, spectral)

        # add two components to contain field and temperature
        self.ncomp = 2
        sizes = tmp_u.shape + (self.ncomp,)

        # invoke super init, passing the communicator and the local dimensions as init
        super().__init__(init=(sizes, comm, tmp_u.dtype))
        self._makeAttributeAndRegister(
            'nvars',
            'eps',
            'radius',
            'spectral',
            'TM',
            'D',
            'dw',
            'L',
            'init_type',
            'comm',
            localVars=locals(),
            readOnly=True,
        )

        L = np.array([self.L] * ndim, dtype=float)

        # get local mesh
        X = list(np.ogrid[self.fft.local_slice(False)])
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = X[i] * L[i] / N[i]
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1.0 / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = list(np.meshgrid(*K, indexing='ij', sparse=True))
        Lp = 2 * np.pi / L
        for i in range(ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)

        # Need this for diagnostics
        self.dx = self.L / nvars[0]
        self.dy = self.L / nvars[1]

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            f.impl[..., 0] = -self.K2 * u[..., 0]

            if self.eps > 0:
                tmp_u = newDistArray(self.fft, False)
                tmp_T = newDistArray(self.fft, False)
                tmp_u = self.fft.backward(u[..., 0], tmp_u)
                tmp_T = self.fft.backward(u[..., 1], tmp_T)
                tmpf = -2.0 / self.eps**2 * tmp_u * (1.0 - tmp_u) * (1.0 - 2.0 * tmp_u) - 6.0 * self.dw * (
                    tmp_T - self.TM
                ) / self.TM * tmp_u * (1.0 - tmp_u)
                f.expl[..., 0] = self.fft.forward(tmpf)

            f.impl[..., 1] = -self.D * self.K2 * u[..., 1]
            f.expl[..., 1] = f.impl[..., 0] + f.expl[..., 0]

        else:
            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = -self.K2 * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])

            if self.eps > 0:
                f.expl[..., 0] = -2.0 / self.eps**2 * u[..., 0] * (1.0 - u[..., 0]) * (1.0 - 2.0 * u[..., 0])
                f.expl[..., 0] -= 6.0 * self.dw * (u[..., 1] - self.TM) / self.TM * u[..., 0] * (1.0 - u[..., 0])

            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = -self.D * self.K2 * u_hat
            f.impl[..., 1] = self.fft.backward(lap_u_hat, f.impl[..., 1])
            f.expl[..., 1] = f.impl[..., 0] + f.expl[..., 0]

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        if self.spectral:
            me = self.dtype_u(self.init)
            me[..., 0] = rhs[..., 0] / (1.0 + factor * self.K2)
            me[..., 1] = rhs[..., 1] / (1.0 + factor * self.D * self.K2)

        else:
            me = self.dtype_u(self.init)
            rhs_hat = self.fft.forward(rhs[..., 0])
            rhs_hat /= 1.0 + factor * self.K2
            me[..., 0] = self.fft.backward(rhs_hat)
            rhs_hat = self.fft.forward(rhs[..., 1])
            rhs_hat /= 1.0 + factor * self.D * self.K2
            me[..., 1] = self.fft.backward(rhs_hat)

        return me

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        def circle():
            tmp_me = newDistArray(self.fft, self.spectral)
            r2 = (self.X[0] - 0.5) ** 2 + (self.X[1] - 0.5) ** 2
            if self.spectral:
                tmp = 0.5 * (1.0 + np.tanh((self.radius - np.sqrt(r2)) / (np.sqrt(2) * self.eps)))
                tmp_me[:] = self.fft.forward(tmp)
            else:
                tmp_me[:] = 0.5 * (1.0 + np.tanh((self.radius - np.sqrt(r2)) / (np.sqrt(2) * self.eps)))
            return tmp_me

        def circle_rand():
            tmp_me = newDistArray(self.fft, self.spectral)
            ndim = len(tmp_me.shape)
            L = int(self.L)
            # get random radii for circles/spheres
            np.random.seed(1)
            lbound = 3.0 * self.eps
            ubound = 0.5 - self.eps
            rand_radii = (ubound - lbound) * np.random.random_sample(size=tuple([L] * ndim)) + lbound
            # distribute circles/spheres
            tmp = newDistArray(self.fft, False)
            if ndim == 2:
                for i in range(0, L):
                    for j in range(0, L):
                        # build radius
                        r2 = (self.X[0] + i - L + 0.5) ** 2 + (self.X[1] + j - L + 0.5) ** 2
                        # add this blob, shifted by 1 to avoid issues with adding up negative contributions
                        tmp += np.tanh((rand_radii[i, j] - np.sqrt(r2)) / (np.sqrt(2) * self.eps)) + 1
            # normalize to [0,1]
            tmp *= 0.5
            assert np.all(tmp <= 1.0)
            if self.spectral:
                tmp_me[:] = self.fft.forward(tmp)
            else:
                tmp_me[:] = tmp[:]
            return tmp_me

        def sine():
            tmp_me = newDistArray(self.fft, self.spectral)
            if self.spectral:
                tmp = 0.5 * (2.0 + 0.0 * np.sin(2 * np.pi * self.X[0]) * np.sin(2 * np.pi * self.X[1]))
                tmp_me[:] = self.fft.forward(tmp)
            else:
                tmp_me[:] = 0.5 * (2.0 + 0.0 * np.sin(2 * np.pi * self.X[0]) * np.sin(2 * np.pi * self.X[1]))
            return tmp_me

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.init_type == 'circle':
            me[..., 0] = circle()
            me[..., 1] = sine()
        elif self.init_type == 'circle_rand':
            me[..., 0] = circle_rand()
            me[..., 1] = sine()
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.init_type)

        return me


# class allencahn_temp_imex_timeforcing(allencahn_temp_imex):
#     """
#     Example implementing Allen-Cahn equation in 2-3D using mpi4py-fft for solving linear parts, IMEX time-stepping,
#     time-dependent forcing
#     """
#
#     def eval_f(self, u, t):
#         """
#         Routine to evaluate the RHS
#
#         Args:
#             u (dtype_u): current values
#             t (float): current time
#
#         Returns:
#             dtype_f: the RHS
#         """
#
#         f = self.dtype_f(self.init)
#
#         if self.spectral:
#
#             f.impl = -self.K2 * u
#
#             tmp = newDistArray(self.fft, False)
#             tmp[:] = self.fft.backward(u, tmp)
#
#             if self.eps > 0:
#                 tmpf = -2.0 / self.eps ** 2 * tmp * (1.0 - tmp) * (1.0 - 2.0 * tmp)
#             else:
#                 tmpf = self.dtype_f(self.init, val=0.0)
#
#             # build sum over RHS without driving force
#             Rt_local = float(np.sum(self.fft.backward(f.impl) + tmpf))
#             if self.comm is not None:
#                 Rt_global = self.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
#             else:
#                 Rt_global = Rt_local
#
#             # build sum over driving force term
#             Ht_local = float(np.sum(6.0 * tmp * (1.0 - tmp)))
#             if self.comm is not None:
#                 Ht_global = self.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
#             else:
#                 Ht_global = Rt_local
#
#             # add/substract time-dependent driving force
#             if Ht_global != 0.0:
#                 dw = Rt_global / Ht_global
#             else:
#                 dw = 0.0
#
#             tmpf -= 6.0 * dw * tmp * (1.0 - tmp)
#             f.expl[:] = self.fft.forward(tmpf)
#
#         else:
#
#             u_hat = self.fft.forward(u)
#             lap_u_hat = -self.K2 * u_hat
#             f.impl[:] = self.fft.backward(lap_u_hat, f.impl)
#
#             if self.eps > 0:
#                 f.expl = -2.0 / self.eps ** 2 * u * (1.0 - u) * (1.0 - 2.0 * u)
#
#             # build sum over RHS without driving force
#             Rt_local = float(np.sum(f.impl + f.expl))
#             if self.comm is not None:
#                 Rt_global = self.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
#             else:
#                 Rt_global = Rt_local
#
#             # build sum over driving force term
#             Ht_local = float(np.sum(6.0 * u * (1.0 - u)))
#             if self.comm is not None:
#                 Ht_global = self.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
#             else:
#                 Ht_global = Rt_local
#
#             # add/substract time-dependent driving force
#             if Ht_global != 0.0:
#                 dw = Rt_global / Ht_global
#             else:
#                 dw = 0.0
#
#             f.expl -= 6.0 * dw * u * (1.0 - u)
#
#         return f
