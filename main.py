from jax import grad, jit
import numpy as np_normal
import jax.numpy as np
from utils import cross_entropy, SGD, sigmoid
from mnist import get_mnist_dataset, LeNet5
from utils import tqdm_
from kernels import construct_phi_for_kernel

# load the dataset
train_loader, test_loader = get_mnist_dataset(batch_size)

# set a batch size for large datasets
batch_size = 64

# define SVGD properties

# - kernel
kernel = 'rbf_random' # alt: 'rbf_analytic', 'rbf', 'rbf_random', 'rbf_matrix', 'hess_matrix' as in kernels.py
kernel_params = {'n_rand_feat': 10}
get_phi = construct_phi_for_kernel(kernel, kernel_params)

# - model
num_particles = 10
theta, forward = LeNet5(
    batch_size=batch_size,
    num_particles=num_particles,
)

@jit
def loss(theta, x, y):
    yhat = forward(theta, x)
    return cross_entropy(y, yhat)

optimizer = SGD(0.001)
grad_loss = jit(grad(loss))


# SVGD training
for epoch in range(100):

    for i, (x, y) in tqdm_(train_loader):
        g = get_phi(theta, grad_loss(theta, x.numpy(), y.numpy()))
        theta = optimizer.update(theta, g)

    test_acc = []
    for i, (x, y) in tqdm_(test_loader):
        yhat = forward(theta, x.numpy())
        nll = cross_entropy(y.numpy(), yhat)
        pred = yhat.mean(axis=0)
        correct = (pred.argmax(axis=1) == y.numpy()).mean()
        test_acc.append(float(correct))

    print("Iteration: ", epoch, "Cross Entropy:", nll, "Test Accuracy:", np_normal.mean(test_acc))
