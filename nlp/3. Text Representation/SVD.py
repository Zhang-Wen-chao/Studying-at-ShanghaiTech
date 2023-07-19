import numpy as np
from sklearn.decomposition import TruncatedSVD

# Generate a random sparse matrix
rng = np.random.RandomState(0)
X = rng.random((6, 8))

# Create an instance of TruncatedSVD
svd = TruncatedSVD(n_components=2)

# Fit and transform the input matrix using SVD
X_transformed = svd.fit_transform(X)

# Print the transformed matrix
print("Transformed Matrix:")
print(X_transformed)