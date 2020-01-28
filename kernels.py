from jax import grad, jit, jacrev, vmap
import numpy as np_normal
import jax.numpy as np
from jax.ops import index, index_update


def get_mapped_kernel_grad(kernel_basic, n_params):
    map_over_a = tuple([0, None] + [None] * n_params)
    map_over_b = tuple([None, 0] + [None] * n_params)

    mv = vmap(kernel_basic, map_over_a, 0)
    kernel = jit(vmap(mv, map_over_b, 1))

    mv = vmap(grad(kernel_basic), map_over_a, 0)
    grad_kernel = jit(vmap(mv, map_over_b, 1))

    return kernel, grad_kernel


### rbf

def exponential_kernel_basic(a, b, h=-1):
    pairwise_dists = -(a - b)
    pairwise_dists_sq = (pairwise_dists ** 2).sum(axis=-1)
    return np.exp(-pairwise_dists_sq / h)


exponential_kernel, grad_exponential_kernel = get_mapped_kernel_grad(exponential_kernel_basic, n_params=1)


def phi(a, rf_0, rf_1, h):
    k = rf_0 @ a
    return np.sqrt(2.0) * np.cos(1.0 / h * k + rf_1)


def random_features_kernel_basic(a, b, rf_0, rf_1, h):
    aa = phi(a, rf_0, rf_1, h)
    bb = phi(b, rf_0, rf_1, h)
    return (aa * bb).mean()


random_features_kernel, grad_random_features_kernel = get_mapped_kernel_grad(random_features_kernel_basic, n_params=3)


def matrix_valued_kernel_basic(a, b, Q, h):
    diff = (a - b)
    return np.linalg.inv(Q) * np.exp(-0.5 / h * diff.T @ Q @ diff)


mv = vmap(matrix_valued_kernel_basic, (0, None, None, None), 0)
matrix_valued_kernel = jit(vmap(mv, (None, 0, None, None), 1))

jac_exp = jacrev(matrix_valued_kernel_basic)


def grad_update(a, b, Q, h, g, repulsion):
    K = matrix_valued_kernel_basic(a, b, Q, h)
    K_der = jac_exp(a, b, Q, h)
    res = np.zeros(a.shape[-1])
    for l in range(res.shape[-1]):
        x = 0
        for m in range(Q.shape[-1]):
            x += K[l, m] * g[m] + repulsion * K_der[l, m, m]
        res = index_update(res, index[l], x)
    return res


mv = vmap(grad_update, (0, None, None, None, 0, None), 0)
grad_update = jit(vmap(mv, (None, 0, None, None, None, None), 1))


# TODO: use jax randomness
def get_random_features_and_params(theta, rfs):
    _, d = theta.shape
    rf_0 = np_normal.random.normal(0, 1, size=(rfs, d))
    rf_1 = np_normal.random.uniform(0, 2 * np.pi, size=(rfs,))
    return rf_0, rf_1


@jit
def median_trick_h(theta):
    pairwise_dists = -((theta[:, np.newaxis, :] - theta))
    pairwise_dists_sq = (pairwise_dists ** 2).sum(axis=-1)
    med_sq = np.median(np.sqrt(pairwise_dists_sq))
    h = med_sq ** 2 / np.log(theta.shape[0] + 1)
    return h


@jit
def rbf_analytic(theta, g, h=None):
    n_particles = theta.shape[0]
    h = h if h is not None else median_trick_h(theta)
    pairwise_dists = -((theta[:, np.newaxis, :] - theta))
    K = np.exp(-(pairwise_dists ** 2).sum(axis=-1) / h)
    grad_K = pairwise_dists * K.reshape((n_particles, n_particles, 1))
    # todo: somehow this line got lost in another version and then whole thing works better without it...why?
    # grad_K = (grad_K * (2.0 / h)).sum(0)
    grad_theta = np.mean(np.matmul(K, g) + grad_K, axis=0)
    return grad_theta


@jit
def rbf(theta, g, h=None):
    h = h if h is not None else median_trick_h(theta)
    K = exponential_kernel(theta, theta, h)
    grad_K = grad_exponential_kernel(theta, theta, h).sum(0)
    grad_theta = np.mean(np.matmul(K, g) + grad_K, axis=0)
    return grad_theta


def rbf_random(theta, g, n_rand_feat=None, h=None):
    h = h if h is not None else median_trick_h(theta)
    n_random_features = n_rand_feat if n_rand_feat is not None else theta.shape[0]
    rf_0, rf_1 = get_random_features_and_params(theta, n_random_features)  # different every iteration!!
    K = random_features_kernel(theta, theta, rf_0, rf_1, h)
    grad_K = grad_random_features_kernel(theta, theta, rf_0, rf_1, h).sum(0)
    grad_theta = np.mean(np.matmul(K, g) + grad_K, axis=0)
    return grad_theta


@jit
def rbf_matrix(theta, g, h=None):
    h = h if h is not None else median_trick_h(theta)
    Q = np.identity(theta.shape[-1])
    grad_theta = grad_update(theta, theta, Q, h, g, 1.0).mean(0)
    return grad_theta


@jit
def hess_matrix(theta, g, hess_logprob, h=None):
    h = h if h is not None else median_trick_h(theta)
    Q = -1.0 * hess_logprob(theta).mean(0)
    grad_theta = grad_update(theta, theta, Q, h, g, 1.0).mean(0)
    return grad_theta


get_phis = {'rbf_analytic': rbf_analytic,
            'rbf': rbf,
            'rbf_random': rbf_random,
            'rbf_matrix': rbf_matrix,
            'hess_matrix': hess_matrix}


def construct_phi_for_kernel(kernel, kernel_params):
    return lambda theta, grad_logprob: get_phis[kernel](theta, grad_logprob, **kernel_params)
