import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

def admm_lasso(X, y, F, lambda_, w0, rho=1, mu=1.01, max_iter=1000, tol=1e-4):
    x = w0.copy()
    z = F @ x
    y_ = np.zeros_like(z)
    
    for _ in range(max_iter):
        x_new = np.linalg.solve(X.T @ X + rho * F.T @ F, X.T @ y + rho * F.T @ (z - y_ / rho))
        z_new = soft_threshold(F @ x_new + y_ / rho, lambda_ / rho)
        y_new = y_ + rho * (F @ x_new - z_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x, z, y_ = x_new, z_new, y_new
        rho *= mu
    
    return x

# Load data
df_train = pd.read_table("/home/zwc/Documents/CS282/CS282-2023-spring/homework/hw5/crime-test.txt")
df_test = pd.read_table("/home/zwc/Documents/CS282/CS282-2023-spring/homework/hw5/crime-test.txt")

y_train = df_train['ViolentCrimesPerPop'].values
X_train = df_train.drop('ViolentCrimesPerPop', axis=1).values

y_test = df_test['ViolentCrimesPerPop'].values
X_test = df_test.drop('ViolentCrimesPerPop', axis=1).values

F = np.identity(X_train.shape[1])

lambdas = np.logspace(-3, 1, 50)

# 2. A plot of log(λ) against the squared error in the 10-folder splited training data.
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_errors = []

for lambda_ in lambdas:
    cv_errors = []
    for train_index, val_index in kf.split(X_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
        w = admm_lasso(X_train_cv, y_train_cv, F, lambda_, np.zeros(X_train.shape[1]))
        cv_errors.append(np.mean((y_val_cv - X_val_cv @ w)**2))
    train_errors.append(np.mean(cv_errors))

plt.plot(np.log(lambdas), train_errors)
plt.xlabel("log(lambda)")
plt.ylabel("Squared Error (10-folder CV)")
plt.title("Squared Error vs log(lambda) in 10-folder Cross-Validation")
plt.show()

# 3. A plot of log(λ) against the squared error in the test data.
test_errors = []

for lambda_ in lambdas:
    w = admm_lasso(X_train, y_train, F, lambda_, np.zeros(X_train.shape[1]))
    test_errors.append(np.mean((y_test - X_test @ w)**2))

plt.plot(np.log(lambdas), test_errors)
plt.xlabel("log(lambda)")
plt.ylabel("Squared Error (Test Data)")
plt.title("Squared Error vs log(lambda) in Test Data")
plt.show()

# 4. A plot of λ against the number of small coefficients
threshold = 1e-4
coef_counts = []

for lambda_ in lambdas:
    w = admm_lasso(X_train, y_train, F, lambda_, np.zeros(X_train.shape[1]))
    coef_counts.append(np.sum(np.abs(w) <= threshold))

plt.plot(lambdas, coef_counts)
plt.xlabel("lambda")
plt.ylabel("Number of Small Coefficients")
plt.title("Number of Small Coefficients vs lambda")
plt.show()

'''
The task of selecting $\lambda$ is crucial in Lasso regularization, as it determines the balance between the model's complexity and its ability to fit the data. A larger $\lambda$ value will result in more coefficients being forced to zero, leading to a simpler model with fewer features. On the other hand, a smaller $\lambda$ value will allow more coefficients to be non-zero, potentially overfitting the data.

In practice, cross-validation can help in selecting an appropriate value for $\lambda$. By plotting the number of small coefficients against $\lambda$, we can observe how the model complexity changes with different regularization strengths. It's essential to choose a $\lambda$ value that results in good test set performance while maintaining a balance between model complexity and the risk of overfitting.

'''




# 5. For the λ that gave the best test set performance, which variable had the largest (most positive) coefficient? What about the most negative? Discuss briefly.
best_lambda = lambdas[np.argmin(test_errors)]
w_best = admm_lasso(X_train, y_train, F, best_lambda, np.zeros(X_train.shape[1]))

max_coef_idx = np.argmax(w_best)
min_coef_idx = np.argmin(w_best)

column_names = df_train.columns[1:]
max_coef_name = column_names[max_coef_idx]
min_coef_name = column_names[min_coef_idx]

print(f"Best lambda: {best_lambda}")
print(f"Largest (most positive) coefficient: {max_coef_name} ({w_best[max_coef_idx]})")
print(f"Most negative coefficient: {min_coef_name} ({w_best[min_coef_idx]})")

'''
[Running] python -u "/home/zwc/Documents/CS282/CS282-2023-spring/homework/hw5/cross.py"
libGL error: MESA-LOADER: failed to open crocus: /usr/lib/dri/crocus_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: crocus
libGL error: MESA-LOADER: failed to open crocus: /usr/lib/dri/crocus_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: crocus
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Best lambda: 0.001
Largest (most positive) coefficient: PctHousOwnOcc (0.5896243875126825)
Most negative coefficient: TotalPctDiv (-0.8503818895472509)

[Done] exited with code=0 in 166.278 seconds

For the value of lambda that gives the best test set performance, we find the variable with the largest (most positive) coefficient and the variable with the smallest (most negative) coefficient. These variables can have a large impact on crime rates. In the real world, however, we need deeper domain knowledge and more background information to explain these coefficients and how they affect crime rates. In practical applications, it is very important to choose an appropriate value of λ, because it can balance the complexity and predictive performance of the model. Through cross-validation, we can find an appropriate value of λ to strike a balance between overfitting and underfitting.
'''