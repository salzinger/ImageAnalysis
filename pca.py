import numpy as np
from sklearn import decomposition


def pca_extract_noise(im, n_components=None):
    im_shape = im[0].shape
    im_flat = np.reshape(im, (im.shape[0], -1))
    pca = decomposition.PCA(n_components=n_components)
    comp = pca.fit_transform(im_flat)
    projected = pca.inverse_transform(comp)
    res = im_flat-projected
    res = res.reshape((-1, *im_shape))
    return res


def split_signal_noise(im, n_components=None, return_residual=True):
    """

    Args:
        return_residual (bool): whether to return the residual

        im: input array of single_image with shape (index-dim, x-dim,  y-dim)
        n_components: Number of components to keep, if n_components is not set all components are kept

    Returns:
        projected: projection of the input array on the subspace spanned by the first n_components, same shape as input
        res: residual of the projection, same shape as input

    """
    im_shape = im[0].shape
    im_flat = np.reshape(im, (im.shape[0], -1))
    pca = decomposition.PCA(n_components=n_components)
    comp = pca.fit_transform(im_flat)
    projected = pca.inverse_transform(comp)
    res = im_flat-projected
    res = res.reshape((-1, *im_shape))
    projected = projected.reshape((-1, *im_shape))

    if return_residual:
        return projected, res
    else:
        return projected


def fit_with_pc(im, n_components=None):
    """

    Args:
        im: input array of single_image with shape (index-dim, x-dim,  y-dim)
        n_components: Number of components to keep, if n_components is not set all components are kept

    Returns:
        projected: projection of the input array on the subspace spanned by the first n_components, same shape as input

    """
    im_shape = im[0].shape
    im_flat = np.reshape(im, (im.shape[0], -1))
    pca = decomposition.PCA(n_components=n_components)
    comp = pca.fit_transform(im_flat)
    projected = pca.inverse_transform(comp)
    projected = projected.reshape((-1, *im_shape))
    return projected


def decompose_images(im, n_components=None):
    """

    Args:
        im: input array of single_image with shape (index-dim, x-dim,  y-dim)
        n_components: Number of components to keep, if n_components is not set all components are kept

    Returns:
        comp: array of components with the shape (index-dim, x-dim,  y-dim)

    """
    im_shape = im[0].shape
    im_flat = np.reshape(im, (im.shape[0], -1))
    pca = decomposition.PCA(n_components=n_components)
    pca = pca.fit(im_flat)
    comp = pca.components_.reshape((-1, *im_shape))
    return comp

