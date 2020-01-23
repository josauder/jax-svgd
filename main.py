from jax import grad, jit
import numpy as np_normal
import jax.numpy as np
from utils import kernel_matrix_and_grad, cross_entropy, get_phi, SGD, sigmoid
from mnist import get_mnist_dataset, LeNet5
from utils import tqdm_

num_particles = 2
batch_size = 16

train_loader, test_loader = get_mnist_dataset(batch_size)
theta, forward = LeNet5(
    batch_size = batch_size,
    num_particles = num_particles,
)

@jit
def loss(theta, x, y):
    yhat = forward(theta, x)
    return cross_entropy(y, yhat)

optimizer = SGD(0.001)
grad_loss = jit(grad(loss))

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
