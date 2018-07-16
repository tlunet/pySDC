from __future__ import division

import numpy as np
from scipy.fftpack import fft, ifft

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.core.Errors import TransferError


class mesh_to_mesh_fft(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between nd meshes with FFT for periodic boundaries

    Attributes:
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """
        # invoke super initialization
        super(mesh_to_mesh_fft, self).__init__(fine_prob, coarse_prob, params)

        self.ratio = int(self.fine_prob.params.nvars / self.coarse_prob.params.nvars)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(self.coarse_prob.init, val=0.0)
            G.values = F.values.flatten()[::self.ratio].reshape(self.coarse_prob.init)
        elif isinstance(F, rhs_imex_mesh):
            G = rhs_imex_mesh(self.coarse_prob.init, val=0.0)
            G.impl.values = F.impl.values.flatten()[::self.ratio].reshape(self.coarse_prob.init)
            G.expl.values = F.expl.values.flatten()[::self.ratio].reshape(self.coarse_prob.init)
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, mesh):
            F = mesh(self.fine_prob.init)
            tmpG = fft(G.values)
            tmpF = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF[0: halfG] = tmpG[0: halfG]
            tmpF[self.fine_prob.init - halfG:] = tmpG[halfG:]
            F.values = np.real(ifft(tmpF))
        elif isinstance(G, rhs_imex_mesh):
            F = rhs_imex_mesh(G)
            tmpG_impl = fft(G.impl.values)
            tmpF_impl = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF_impl[0: halfG] = tmpG_impl[0: halfG]
            tmpF_impl[self.fine_prob.init - halfG:] = tmpG_impl[halfG:]
            tmpG_expl = fft(G.expl.values)
            tmpF_expl = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF_expl[0: halfG] = tmpG_expl[0: halfG]
            tmpF_expl[self.fine_prob.init - halfG:] = tmpG_expl[halfG:]
            F.impl.values = np.real(ifft(tmpF_impl))
            F.expl.values = np.real(ifft(tmpF_expl))
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
