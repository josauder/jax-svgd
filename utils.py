from jax import grad, jit, random, vmap
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

@jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#TODO remove hardcoded 10
@jit
def to_one_hot(y):
    return np.eye(10)[y]

@jit
def cross_entropy(y, yhat):
    y = to_one_hot(y)
    y = y.reshape(1, y.shape[0], y.shape[1])
    return -np.mean(np.sum(y * yhat, axis=2))


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

    def predict_one(theta,x):
        rng_key=random.PRNGKey(0)
        theta_flat = unflattener(theta)
        return predict(theta_flat, x, rng=rng_key)

    predict_batch = jit(vmap(predict_one,(0,None),0))

    return thetas, predict_batch


def tqdm_(loader):
    return enumerate(tqdm(loader, ascii=True, leave=False))

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, theta, grad):
        return theta - self.learning_rate * grad
