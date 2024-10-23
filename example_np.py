import numpy as np
import itertools


dist_inf = 10
d_a = np.array([
    [0, 2, 6, 3],
    [2, 0, 4, 5],
    [6, 4, 0, dist_inf],
    [3, 5, dist_inf, 0]
])

d_b = np.array([
    [0, dist_inf, 4, 6],
    [dist_inf, 0, 5, 3],
    [4, 5, 0, 2],
    [6, 3, 2, 0]
])

gt_p = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.int32)


def permutation_to_matrix(perm, N):
    # Create an identity matrix
    identity = np.eye(N)
    # Rearrange rows according to the permutation
    perm_matrix = identity[perm, :]
    return perm_matrix


def test_gt():
        
    edge_err1 = d_a - gt_p @ d_b @ gt_p.T
    edge_err1 = np.sum(edge_err1 ** 2)

    edge_err2 = 2 * np.trace(gt_p.T @ d_a.T @ gt_p @ d_b) - np.trace(d_a.T @ d_a) - np.trace(d_b.T @ d_b)

    print('Edge errors should be zeros in both cases:', edge_err1, edge_err2)


def sqrt_decomposition(d):
    u, sigma, vt = np.linalg.svd(d)
    s = np.diag(np.sqrt(sigma))
    return u @ s @ vt


def clap(d_a, d_b):

    row_sum_a = np.sum(d_a, axis=1)
    row_sum_b = np.sum(d_b, axis=1)
    row_sum_max_a = np.max(row_sum_a)
    row_sum_max_b = np.max(row_sum_b)
    row_sum_max = max(row_sum_max_a, row_sum_max_b)

    n = d_a.shape[0]
    d_a_ = d_a.copy()
    d_b_ = d_b.copy()
    for i in range(n):
        d_a_[i, i] = row_sum_max
        d_b_[i, i] = row_sum_max

    h_a = sqrt_decomposition(d_a_)
    h_b = sqrt_decomposition(d_b_)

    return h_a, h_b


def eval_clap():

    h_a, h_b = clap(d_a, d_b)

    n = d_a.shape[0]

    edge_gt = h_a.T @ gt_p @ h_b
    edge_term_gt = np.abs(edge_gt).sum()
    quad_term_gt = np.square(edge_gt).sum()

    print('edge abs term gt:', edge_term_gt, 'edge quad term gt:', quad_term_gt)
    permutations = list(itertools.permutations(np.arange(n)))

    err_count = 0
    for perm in permutations:
        perm_mat = permutation_to_matrix(perm, n)
        edge_perm = h_a.T @ perm_mat @ h_b
        edge_term_perm = np.abs(edge_perm).sum()
        quad_term_perm = np.square(edge_perm).sum()

        print('\tedge abs term perm:', edge_term_perm, 'edge quad term perm:', quad_term_perm)
        if edge_term_perm > edge_term_gt:
            print('Error in the clap approximation:', perm)
            err_count += 1

    if err_count == 0:
        print('Clap approximation is correct')


if __name__ == '__main__':

    test_gt()
    eval_clap()
