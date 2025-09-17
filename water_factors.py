"""
Copyright (c) 2025 Paul Irofti <paul@irofti.net>
Copyright (c) 2025 Luis Romero-Ben <luis.romero.ben@upc.edu>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

# import time
from types import SimpleNamespace
from typing import List

import gtsam
import numpy as np
from gtsam import CustomFactor


# Define RMSE function
def rmse(est, true):
    return np.sqrt(np.mean((est - true) ** 2))


def build_pipe(E: np.ndarray, N: int) -> np.ndarray:
    pipe = np.empty([N, N])
    for i in range(N):  # we have 4 measurements
        for j in range(N):  # we have 10 nodal heads in h
            # we find the index of the pipe connecting nodes i and j
            pipe[i, j] = find_pair_index_unordered(E, (i, j))
    pipe = pipe.astype(int)
    return pipe


def find_pair_index_unordered(E, target_pair):
    target_set = set(target_pair)
    for i, row in enumerate(E):
        if set(row) == target_set:
            return i
    return -1


def compute_B(h: np.ndarray, E: np.ndarray) -> np.ndarray:
    # required_matrices = loadmat("test_data_complete_leak88_Modena.mat")
    # E2 = required_matrices["E"]
    N = h.shape[0]
    M = E.shape[0]
    B = np.zeros((M, N))

    for i in range(M):
        a, b = (
            E[i, 0],
            E[i, 1],
        )  # -1 to convert MATLAB indices to Python indices
        a, b = int(a), int(b)  # in case E is float-typed
        if h[a] >= h[b]:
            B[i, a] = 1
            B[i, b] = -1
        else:
            B[i, b] = 1
            B[i, a] = -1

    return B.T


def compute_GandJ(
    h: np.ndarray, E: np.ndarray, pipe_props: SimpleNamespace
) -> tuple[np.ndarray, np.ndarray]:
    N = h.shape[0]
    M = E.shape[0]
    G = np.zeros((N, N))
    J = np.zeros((N, N))  # we call J to the node-node incidence matrix
    for i in range(M):
        a, b = (
            E[i, 0],
            E[i, 1],
        )  # -1 to convert MATLAB indices to Python indices
        a, b = int(a), int(b)
        # Compute G
        G[a, b] = (
            (pipe_props.roughness[i] ** 1.852) * (pipe_props.diameters[i] ** 4.8704)
        ) / (10.675 * pipe_props.lengths[i])
        G[b, a] = G[a, b]
        # Compute J
        if h[a] >= h[b]:
            J[a, b] = 1
            J[b, a] = -1
        else:
            J[a, b] = -1
            J[b, a] = 1
    return G, J


def compute_AW_mats(
    h: np.ndarray, E: np.ndarray, G: np.ndarray, J: np.ndarray, hw_exp: float
) -> tuple[np.ndarray, np.ndarray]:
    N = h.shape[0]
    M = E.shape[0]
    Omega = np.zeros((N, N))
    for i in range(M):
        a, b = (
            E[i, 0],
            E[i, 1],
        )  # -1 to convert MATLAB indices to Python indices
        a, b = int(a), int(b)
        # Compute Omega
        Omega[a, b] = (G[a, b] ** hw_exp) * (J[a, b] * (h[a] - h[b])) ** (hw_exp - 1)
        Omega[b, a] = Omega[a, b]
    Phi = np.diag(np.sum(Omega, axis=0))
    return Omega, Phi


def obtain_ps(
    h: np.ndarray,
    E: np.ndarray,
    pipe_props: SimpleNamespace,
    Ss: np.ndarray,
    hw_exp: float,
) -> np.ndarray:
    G, J = compute_GandJ(h, E, pipe_props)
    Omega, Phi = compute_AW_mats(h, E, G, J, hw_exp)
    Phim2 = np.linalg.inv(Phi @ Phi)
    Ld = (Phi - Omega).T @ Phim2 @ (Phi - Omega)
    return np.linalg.inv(0.1 * Ld + Ss.T @ Ss) @ Ss.T


def error_func(this: CustomFactor, v: gtsam.Values, H: List[np.ndarray]) -> np.ndarray:
    key = this.keys()[0]
    key1 = this.keys()[1]
    h = v.atVector(key)
    d = v.atVector(key1)

    invT = error_func.invT
    E = error_func.E
    pipe = error_func.pipe
    # start = time.time()

    hw_exp = 1 / 1.852
    B = compute_B(h, E)
    Bt = B.T

    # Error function

    d_model = -B @ (invT @ B.T @ h) ** hw_exp
    test = B.T @ h
    if np.any(test < 0):
        print("HERE")
    error = d - d_model
    # print(error)

    # We compute the Jacobian by considering the derivative of d_v with respect to h

    Jd = np.zeros((len(d), len(h)))
    for i in range(len(d)):
        for j in range(len(h)):
            # k will only be -1 if i and j are not connected, in that case Jd[i,j] = 0
            if i == j:
                for u in range(len(h)):
                    k2 = pipe[u, j]
                    if k2 > -1:
                        Jd[i, j] = (
                            Jd[i, j]
                            + -B[i, k2]
                            * (
                                hw_exp
                                * (invT[k2, k2] * (Bt[k2, u] * h[u] + Bt[k2, j] * h[j]))
                                ** (hw_exp - 1)
                            )
                            * invT[k2, k2]
                            * Bt[k2, j]
                        )
                    # if u == 232:
                    #     print('HERE u')
            else:
                k = pipe[i, j]
                if k > -1:
                    Jd[i, j] = (
                        -B[i, k]
                        * (
                            hw_exp
                            * (invT[k, k] * (Bt[k, i] * h[i] + Bt[k, j] * h[j]))
                            ** (hw_exp - 1)
                        )
                        * invT[k, k]
                        * Bt[k, j]
                    )
            # if j == 661:
            #     print('HERE j')
        # if i == 232:
        #     print('HERE i')
    # stop = time.time()

    error_func.counter += 1
    inf_indices = np.argwhere(np.isinf(Jd))
    for idx in inf_indices:
        print(f"Fila {idx[0]}, Columna {idx[1]}, Valor: {Jd[idx[0], idx[1]]}")
    if np.isinf(Jd).any():
        print("inf")
    # print(f"{error_func.counter} [{stop - start}]")
    if H is not None:
        H[0] = -Jd
        H[1] = np.eye(len(d))
    return error


def error_localization(
    this: CustomFactor, v: gtsam.Values, H: List[np.ndarray]
) -> np.ndarray:
    key = this.keys()[0]
    key1 = this.keys()[1]
    key2 = this.keys()[2]
    h = v.atVector(key)
    d = v.atVector(key1)
    ll = v.atVector(key2)

    # Fetch extra arguments
    invT = error_localization.invT
    E = error_localization.E
    pipe = error_localization.pipe
    dh_n = error_localization.dh_n
    d_n = error_localization.d_n

    # start = time.time()

    hw_exp = 1 / 1.852
    B = compute_B(h, E)
    Bt = B.T

    # Error function
    d_model = -B @ (invT @ B.T @ h) ** hw_exp
    error = ll - d - d_model + dh_n + d_n
    # print(error)

    # We compute the Jacobian by considering the derivative of d_v with respect to h

    Jd = np.zeros((len(d), len(h)))
    for i in range(len(d)):
        for j in range(len(h)):
            # k will only be -1 if i and j are not connected, in that case Jd[i,j] = 0
            if i == j:
                for u in range(len(h)):
                    k2 = pipe[u, j]
                    if k2 > -1:
                        Jd[i, j] = (
                            Jd[i, j]
                            + -B[i, k2]
                            * (
                                hw_exp
                                * (invT[k2, k2] * (Bt[k2, u] * h[u] + Bt[k2, j] * h[j]))
                                ** (hw_exp - 1)
                            )
                            * invT[k2, k2]
                            * Bt[k2, j]
                        )
            else:
                k = pipe[i, j]
                if k > -1:
                    Jd[i, j] = (
                        -B[i, k]
                        * (
                            hw_exp
                            * (invT[k, k] * (Bt[k, i] * h[i] + Bt[k, j] * h[j]))
                            ** (hw_exp - 1)
                        )
                        * invT[k, k]
                        * Bt[k, j]
                    )

    # stop = time.time()

    error_localization.counter += 1
    # print(f"{error_localization.counter} [{stop - start}]")

    if H is not None:
        H[0] = -Jd
        H[1] = -np.eye(len(d))
        H[2] = np.eye(len(ll))
    return error
