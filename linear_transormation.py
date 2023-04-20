import numpy as np

def lin_map_matrix(FV, W):
    """For ß basis in V and γ basis in W,
    returns with the [T]^γ_ß matrix of the transformation T,
    which transforms v in V to fv in FV"""
    return np.transpose([np.linalg.solve(np.transpose(W), fv) for fv in FV])


# def get_basis(M):
#     # Define the vectors as rows of a matrix
#     vectors = M
#
#     # Compute the rank of the matrix
#     rank = np.linalg.matrix_rank(vectors)
#
#     # Check if the vectors are linearly independent
#     if rank == vectors.shape[0]:
#         # The vectors are linearly independent, so they form a basis
#         basis = vectors
#     else:
#         # The vectors are linearly dependent, so we need to find a subset that forms a basis
#         # One way to do this is to perform row reduction on the matrix and select the rows corresponding to the pivot columns
#         rref, pivots = np.linalg.qr(vectors, mode='r')
#         basis = vectors[pivots.astype(int), :]  # convert pivots to integer type
#
#     return basis


def linmap_predict(v, M, V, W):
    return np.matmul(np.transpose(W), np.matmul(M, np.linalg.solve(np.transpose(V), v)))


def get_basis(vectors):
    return np.linalg.qr(vectors)[0]


# v1 = [1237, 723]
# v2 = [361, 658]
#
#
# w1 = [1869, 1657]
# w2 = [2022, 2216]

# V = np.array([v1, v2])
# V = get_basis(V).transpose()  # V = V.reshape((3, 2))
#
# c1_v1 = np.array([1869, 2022])
# c2_v1 = np.array([1657, 2216])
# W = np.array([c1_v1, c2_v1])
# W = get_basis(W)
# W_inv = np.linalg.inv(W)
#
# v_V = np.array([1237, 361])
# v_W = np.dot(W_inv, np.dot(V.transpose(), v_V))
# v_W_reshape = v_W.reshape((2, 1))
# v_W_reshape = np.reshape(v_W, (2,1))
# v_real = W @ v_W_reshape
# v_real_int = np.round(v_real).astype(int)
# # coeffs = np.linalg.solve(np.array([[c1_v1[0], c2_v1[0]], [c1_v1[1], c2_v1[1]]]), v_W)
# # v = coeffs[0] * c1_v1 + coeffs[1] * c2_v1 @ v_W_reshape
# print(v_real)

#
# v1 = np.array([1237, 361])
# v2 = np.array([723, 658])
# B = np.column_stack((v1, v2))
# if np.linalg.det(B) == 0:
#     print("B not a base")
#
# # Define the new basis
# w1 = np.array([1869, 2022])
# w2 = np.array([1657, 2216])
# C = np.column_stack((w1, w2))
#
# if np.linalg.det(C) == 0:
#     print("C not a base")
#
# # Define the vector v in the original basis
# v = np.array([0, 1]).reshape((-1, 1))
#
# # Find the coordinates of v in the standard basis
# v_standard = np.dot(B, v)
#
# # Find the coordinates of v in the new basis
# v_C = np.dot(np.linalg.inv(C), v_standard)
# # v_C = np.dot(v_out, v)
#
# # Print the result
# print("The coordinate vector v in the basis C is: ")
# print(v_C.flatten())
# v_out = v_C[0] * w1 + v_C[1] * w2
# print(v_out)

# Define the original basis vectors
v1 = np.array([1237, 361])
v2 = np.array([723, 658])

# Define the new basis vectors
w1 = np.array([1869, 2022])
w2 = np.array([1657, 2216])

# Create the transformation matrix from the original to the new basis
M = np.column_stack((w1, w2))
M_inv = np.linalg.inv(M)
N = np.column_stack((v1, v2))
T = np.dot(M_inv, N)

# Create the homogeneous transformation matrix
T_homo = np.identity(3)
T_homo[:2, :2] = T

# Define the vector in the original basis
v = np.array([1237, 361, 1])

# Transform the vector to the new basis
v_new = np.dot(T_homo, v)
x_new = v_new[0] / v_new[2]
y_new = v_new[1] / v_new[2]
v_output = [x_new, y_new]
# Print the result
print("The vector in the new basis is: ", v_output)