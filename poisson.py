import numpy as np
import torch as th

from numpy import newaxis as na


# Utils

def _thfl_to_thcx(t):
    return th.stack([t, th.zeros_like(t)], axis=-1)

def _thcx_to_thfl(t):
    return t[...,0]

def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi



# Sub-functions

def solvePoisson_precomped(rho, scale):
    #rho = _thfl_to_thcx(rho)
    fourier_phi = th.fft.fftn(rho, dim=(0,1,2)) / scale
    phi = th.fft.ifftn(fourier_phi, dim=(0,1,2)).real
    #phi = _thcx_to_thfl(phi)
    return phi

def precomp_Poissonscaling(rho):
    L, M, N = rho.size()
    l, m, n = th.arange(L), th.arange(M), th.arange(N)

    scale = -4 * (th.sin(np.pi * l / L)[:, na, na] ** 2 \
                + th.sin(np.pi * m / M)[na, :, na] ** 2 \
                + th.sin(np.pi * n / N)[na, na, :] ** 2)
    scale[0,0,0] = 1
    scale = scale.to(rho.device)
    return scale

def applyQ(p, WWx, WWy, WWz, wrap=False):
    dx = p[1:,:,:] - p[:-1,:,:]
    dy = p[:,1:,:] - p[:,:-1,:]
    dz = p[:,:,1:] - p[:,:,:-1]

    if (wrap):
        dx = _wrap_to_pi(dx)
        dy = _wrap_to_pi(dy)
        dz = _wrap_to_pi(dz)

    WWdx = WWx * dx
    WWdy = WWy * dy
    WWdz = WWz * dz

    WWdx = WWdx[1:,:,:] - WWdx[:-1,:,:]
    WWdy = WWdy[:,1:,:] - WWdy[:,:-1,:]
    WWdz = WWdz[:,:,1:] - WWdz[:,:,:-1]

    Qp = th.zeros_like(p)
    Qp[1:-1, :, :] += WWdx
    Qp[:, 1:-1, :] += WWdy
    Qp[:, :, 1:-1] += WWdz
    return Qp


# Main function

def poisson_unwrap_gpu(psi, weight=None, kmax=100):
    psi = th.FloatTensor(psi).cuda()
    WW = th.ones_like(psi) if (weight is None) \
                           else th.FloatTensor(weight).cuda() ** 2

    WWx = th.min(WW[1:,:,:], WW[:-1,:,:])
    WWy = th.min(WW[:,1:,:], WW[:,:-1,:])
    WWz = th.min(WW[:,:,1:], WW[:,:,:-1])

    rk = applyQ(psi, WWx, WWy, WWz, wrap=True)
    normR0 = th.norm(rk)

    eps = 1e-6
    phi = th.zeros_like(psi)
    scale = precomp_Poissonscaling(rk)
    for k in range(kmax):
        zk = solvePoisson_precomped(rk, scale)
        rk_zk_sum = (rk * zk).sum()
        pk = zk if (k == 0) else zk + pk * rk_zk_sum / rk_zk_prev_sum
        rk_zk_prev_sum = rk_zk_sum

        Qpk = applyQ(pk, WWx, WWy, WWz)
        alphak = rk_zk_sum / (pk * Qpk).sum()
        phi += alphak * pk
        rk -= alphak * Qpk

        if (th.norm(rk) < eps * normR0):
            break
    return phi.data.cpu().numpy()