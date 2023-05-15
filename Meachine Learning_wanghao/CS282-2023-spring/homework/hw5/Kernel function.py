import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# Define data
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 2], [2, -2], [-2, 2], [-2, -2]])
Y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# Define kernel function
def poly_kernel(x, y):
    return (1 + np.dot(x, y))**3

# Construct kernel matrix
K = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        K[i, j] = poly_kernel(X[i], X[j])

# Define quadratic program
P = matrix(np.outer(Y, Y) * K)
q = matrix(-1 * np.ones((8, 1)))
G = matrix(np.concatenate((-1 * np.eye(8), np.eye(8)), axis=0))
h = matrix(np.concatenate((np.zeros((8, 1)), np.ones((8, 1)) * 1000), axis=0))
A = matrix(Y.reshape((1, 8)), tc='d')
b = matrix(0.0)

# Solve quadratic program
solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# Extract support vectors
threshold = 1e-5
support_indices = np.where(alphas > threshold)[0]
support_vectors = X[support_indices]
support_alphas = alphas[support_indices]
support_labels = Y[support_indices]

# Compute bias term
bias = np.mean(support_labels - np.sum(support_alphas * support_labels * K[:, support_indices], axis=1))

# Create a mesh grid to plot decision boundary
x_min, x_max = -3, 3
y_min, y_max = -3, 3
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
xy = np.c_[xx.ravel(), yy.ravel()]

# Compute decision boundary
zz = np.array([np.sum(support_alphas * support_labels * np.array([poly_kernel(x, y) for y in support_vectors])) for x in xy]).reshape(xx.shape) + bias

# Plot data points and decision boundary
plt.figure(figsize=(5, 5))
plt.contour(xx, yy, zz, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors=['blue', 'green', 'red'])
plt.scatter(X[:4, 0], X[:4, 1], marker='D', color='blue', s=50, label='+1')
plt.scatter(X[4:, 0], X[4:, 1], marker='s', color='red', s=50, label='-1')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker='o', color='black', s=100, facecolors='none')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='best')
plt.title('Decision boundary')
plt.show()
