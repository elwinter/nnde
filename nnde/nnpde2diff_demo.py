import numpy as np

from nnde.nnpde2diff import NNPDE2DIFF
from nnde.pde2diff import PDE2DIFF
from nnde.trainingdata import create_training_grid


# Create training data.
nx = 10
nt = 10
xt_train = np.array(create_training_grid([nx, nt]))

# Options for training
training_opts = {}
training_opts['debug'] = False
training_opts['verbose'] = True
training_opts['eta'] = 0.01
training_opts['maxepochs'] = 1000
training_opts['use_jacobian'] = False
H = 10  # Hidden layer nodes

pde = 'eq.diff1d_halfsine'
print('Examining %s.' % pde)

# Read the equation definition.
eq = PDE2DIFF(pde)

# Fetch the dimensionality of the problem.
m = len(eq.bc)
print('Differential equation %s has %d dimensions.' % (eq, m))

x_train = xt_train
n = len(x_train)

# Analytical solution (if available)
Ya = None
if eq.Ya is not None:
    print("Computing analytical solution at training points.")
    Ya = np.zeros(n)
    for i in range(n):
        Ya[i] = eq.Ya(x_train[i])
    print('Ya =', Ya)

# Analytical gradient (if available)
delYa = None
if eq.delYa is not None:
    print("Computing analytical gradient at training points.")
    delYa = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            delYa[i][j] = eq.delYa[j](x_train[i])
    print('delYa =', delYa)

# Analytical Laplacian (if available)
del2Ya = None
if eq.del2Ya is not None:
    print("Computing analytical Laplacian at training points.")
    del2Ya = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            del2Ya[i][j] = eq.del2Ya[j](x_train[i])
    print('del2Ya =', del2Ya)

# Select training algorithm.
trainalg = 'BFGS'
print('Training using %s algorithm.' % trainalg)

# Create and train the neural network.
net = NNPDE2DIFF(eq, nhid=H)
np.random.seed(0)

# Train the network.
try:
    net.train(x_train, trainalg=trainalg, opts=training_opts)
except (OverflowError, ValueError) as e:
    print('Error using %s algorithm on %s!' % (trainalg, pde))
    print(e)

# If available, print the report from minimize().
if net.res:
    print(net.res)

print('The trained network is:')
print(net)

# Compute the trained results at the training points.
Yt = net.run_debug(x_train)
print('The trained solution is:')
print('Yt =', Yt)

if eq.Ya:
    Yt_err = Yt - Ya
    print('The error in the trained solution is:')
    print('Yt_err =', Yt_err)

# Compute the trained gradient.
delYt = net.run_gradient_debug(x_train)
print('The trained gradient is:')
print('delYt =', delYt)

if eq.delYa:
    delYt_err = delYt - delYa
    print('The error in the trained gradient is:')
    print('delYt_err =', delYt_err)

# Compute the trained Laplacian.
del2Yt = net.run_laplacian_debug(x_train)
print('The trained Laplacian is:')
print('del2Yt =', del2Yt)

if eq.del2Ya:
    del2Yt_err = del2Yt - del2Ya
    print('The error in the trained Laplacian is:')
    print('del2Yt_err =', del2Yt_err)
