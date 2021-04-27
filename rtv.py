import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import factorized


def gradient(img):
    """Compute the spatial gradients (fx, fy) with finite differences"""
    fx = np.diff(img, axis=1, append=img[:,-1:,:])
    fy = np.diff(img, axis=0, append=img[-1:,:,:])

    return fx, fy


def lpfilter(img, sigma):
    """Low pass gaussian filter"""
    result = np.zeros_like(img)
    for c in range(result.shape[2]):
        result[..., c] = gaussian_filter(img[..., c], sigma, mode="constant", truncate=2.5)
    return result


def compute_texture_weights(img, sigma, sharpness):
    fx, fy = gradient(img)
    wto = 1/np.maximum(np.mean(np.sqrt(fx**2 + fy**2), axis=-1), sharpness)

    # Low resolution gradients
    lr_img = lpfilter(img, sigma)
    gfx, gfy = gradient(lr_img)

    wtbx = 1/np.maximum(np.mean(np.abs(gfx), axis=-1), 1e-3)
    wtby = 1/np.maximum(np.mean(np.abs(gfy), axis=-1), 1e-3)

    wx = wtbx * wto
    wy = wtby * wto

    wx[:, -1] = 0
    wy[-1, :] = 0

    return wx, wy


def construct_laplacian(wx, wy, lam):
    H, W = wx.shape

    dx = -lam * np.ravel(wx)
    dy = -lam * np.ravel(wy)

    offsets = np.array([-W, -1])
    A = diags([dy, dx], offsets, shape=(H*W, H*W))

    # Pad
    ddy = np.pad(dy, (W, 0))[:-W]
    ddx = np.pad(dx, (1, 0))[:-1]

    # diagonal values
    D = 1 - (dx + dy + ddx + ddy)
    D = diags(D, 0, shape=(H*W, H*W))

    return D + A + A.T


def solve_linear_equation(img, x, wx, wy, lam):
    """
    The code for constructing inhomogenious Laplacian is adapted from
    the implementation of the wlsFilter.
    The same matrix and weights are used for all channels
    """
    L = construct_laplacian(wx, wy, lam)
    LU = factorized(L)

    output = np.zeros_like(img)

    for c in range(img.shape[2]):
        vectorized = LU(np.ravel(img[..., c]))
        output[..., c] = vectorized.reshape(output[..., c].shape)

    return output


def tsmooth(img, lam=0.01, sigma=3.0, sharpness=0.02, n_iter=4):
    """tsmooth - Structure Extraction from Texture via Relative Total Variation
    S = tsmooth(I, lambda, sigma, maxIter) extracts structure S from
    structure+texture input I, with smoothness weight lambda, scale
    parameter sigma and iteration number maxIter.

    Params:
    @img       : Input UINT8 image, both grayscale and color images are acceptable.
    @lam       : Parameter controlling the degree of smooth.
                 Range (0, 0.05], 0.01 by default.
    @sigma     : Parameter specifying the maximum size of texture elements.
                 Range (0, 6], 3 by defalut.
    @sharpness : Parameter controlling the sharpness of the final results,
                 which corresponds to \epsilon_s in the paper [1]. The smaller the value, the sharper the result.
                 Range (1e-3, 0.03], 0.02 by defalut.
    @n_iter    : Number of iterations, 4 by default.

    ==========
    The Code is created based on the method described in the following paper
    [1] "Structure Extraction from Texture via Relative Total Variation", Li Xu, Qiong Yan, Yang Xia, Jiaya Jia, ACM Transactions on Graphics,
    (SIGGRAPH Asia 2012), 2012.
    The code and the algorithm are for non-comercial use only.
    """
    lam = lam / 2.0
    x = img.copy()

    for _ in range(n_iter):
        wx, wy = compute_texture_weights(x, sigma, sharpness)
        x = solve_linear_equation(img, x, wx, wy, lam)
        sigma = max(sigma / 2.0, 0.5)

    return x


if __name__ == '__main__':
    img = imread("imgs/Bishapur_zan.jpg").astype(float) / 255.0
    smoothed = tsmooth(img, lam=0.015)
    plt.imshow(smoothed)
    plt.show()
