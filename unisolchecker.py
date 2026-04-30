import numpy as np

# Matrix A and vector b
A = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
b = np.array([1, 1, 1])

# Use least squares to find the solution
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print(f"Solution: w1 = {x[0]:.0f}, w2 = {x[1]:.0f}")
print(f"Residuals (0 means exact solution): {residuals}")