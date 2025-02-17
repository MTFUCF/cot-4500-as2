import numpy as np
from scipy.interpolate import CubicSpline

#Neville's Method
def neville(x, y, x_target):
    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x_target - x[i + j]) * Q[i, j - 1] - (x_target - x[i]) * Q[i + 1, j - 1]) / (x[i] - x[i + j])

    return Q[0, -1]

x_vals_1 = np.array([3.6, 3.8, 3.9])
y_vals_1 = np.array([1.675, 1.436, 1.318])
print(neville(x_vals_1, y_vals_1, 3.7), "\n")

#Newton's Forward Difference Method
def newton_forward(x, y):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y  

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x[i + j] - x[i])

    return diff_table[0, 1:4] 

x_vals_2 = np.array([7.2, 7.4, 7.5, 7.6])
y_vals_2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])
print("\n".join(map(str, newton_forward(x_vals_2, y_vals_2))), "\n")

#Newton's Forward Method
def newton_interpolation(x, y, x_target):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x[i + j] - x[i])

    result = y[0]
    term = 1
    for j in range(1, n):
        term *= (x_target - x[j - 1])
        result += term * diff_table[0, j]

    return result

print(newton_interpolation(x_vals_2, y_vals_2, 7.3), "\n")

# 4. Hermite Polynomial Approximation
def hermite_interpolation_fixed(x, y, dy):
    """ Constructs the Hermite interpolation divided difference table correctly. """
    n = len(x)
    H = np.zeros((2 * n, 5))  
 
    for i in range(n):
        H[2 * i][0] = H[2 * i + 1][0] = x[i]  
        H[2 * i][1] = H[2 * i + 1][1] = y[i]  
        H[2 * i + 1][2] = dy[i] 

    for i in range(1, 2 * n, 2):  
        H[i, 2] = dy[i // 2]  
        H[i - 1, 2] = dy[i // 2]  

    for j in range(3, 5):  
        for i in range(2 * n - j):
            denominator = H[i + j - 2, 0] - H[i, 0]
            if abs(denominator) > 1e-12:  # Avoid division by zero
                H[i, j] = (H[i + 1, j - 1] - H[i, j - 1]) / denominator

    return H

print("\n")
x_vals_4 = np.array([3.6, 3.8, 3.9])
y_vals_4 = np.array([1.675, 1.436, 1.318])
dy_vals_4 = np.array([-1.195, -1.188, -1.182])
hermite_table_fixed = hermite_interpolation_fixed(x_vals_4, y_vals_4, dy_vals_4)

for row in hermite_table_fixed:
    print("[", "  ".join(f"{val:.8e}" for val in row), "]")
print("\n")

def cubic_spline_matrix(x, y):
    n = len(x) - 1
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    h = np.diff(x)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    A[0, 0] = A[-1, -1] = 1
    return A, b

x_vals_5 = np.array([2, 5, 8, 10])
y_vals_5 = np.array([3, 5, 7, 9])
A_matrix, b_vector = cubic_spline_matrix(x_vals_5, y_vals_5)
x_vector = np.linalg.solve(A_matrix, b_vector)

for row in A_matrix:
    print(np.array(row))  
print(np.array(b_vector))  
print(np.array(x_vector))  