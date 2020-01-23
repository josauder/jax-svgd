from jax import grad, jit, random
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

@jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#TODO: tune h!
@jit
def kernel_matrix_and_grad(theta, h=0.1):
    n_particles = theta.shape[0]
    pairwise_dists = -((theta[:,np.newaxis,:] - theta))
    K = np.exp(-(pairwise_dists**2).sum(-1) / h**2)
    grad_K = pairwise_dists *  K.reshape((n_particles, n_particles, 1))
    return K, grad_K

#TODO remove hardcoded 10
@jit
def to_one_hot(y):
    return np.eye(10)[y]

@jit
def cross_entropy(y, yhat):
    y = to_one_hot(y)
    y = y.reshape(1, y.shape[0], y.shape[1])
    return -np.mean(np.sum(y * yhat, axis=2))

@jit
def get_phi(theta, g):
    K, grad_K = kernel_matrix_and_grad(theta)
    phi = np.mean(np.matmul(K, g) + grad_K, axis=0)
    return phi


def make_model(model, input_shape, num_particles):
    init_random_params, predict = model
    keys = random.split(random.PRNGKey(0), num_particles)

    params_ = []
    for i in range(num_particles):
        out_shape, params = init_random_params(rng=keys[i], input_shape=input_shape)
        params_.append(params)

    _, unflattener = ravel_pytree(params_[0])

    thetas = np.array([
        np.array(ravel_pytree(params_[i])[0]) for i in range(num_particles)
    ])

    #TODO: use vmap instead of loop!
    @jit
    def predict_(thetas, x, rng_key=random.PRNGKey(0)):
        outs = []
        for i in range(num_particles):
            theta_i = unflattener(thetas[i])
            out = predict(theta_i, x, rng=rng_key)
            outs.append(out)
        return np.array(outs)

    return thetas, predict_


def tqdm_(loader):
    return enumerate(tqdm(loader, ascii=True, leave=False))

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, theta, grad):
        return theta - self.learning_rate * grad
