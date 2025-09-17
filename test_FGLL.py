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

import numpy as np
from scipy.io import loadmat, savemat

from FGLL import WaterFactorGraphEstimator

network = "Modena"  # "LTOWN" # "toy_example"
if network == "Modena":
    matdata = loadmat(
        f"network_data/{network}/test_data_complete_leak_{network}_40DS.mat"
    )
    leakdata = loadmat(f"network_data/{network}/leak_list_{network}.mat")
    leak_list = leakdata["leak_list"] - 1
    leak_complete = leak_list[0]
    n_leaks = len(leak_complete)
    n_times = 75
    noise_cov = [1e-12, 1e-4, 1e-12, 1e-9, 1e-12, 1e-3, 1e-4, 1e-9]
elif network == "LTOWN":
    leak_complete = [461, 232, 628, 538, 866, 183, 158, 369]
    matdata = loadmat(
        f"network_data/{network}/test_data_{leak_complete[0]}_leak_{network}_delpump.mat"
    )
    n_leaks = len(leak_complete)
    n_times = 288
    noise_cov = [1e-12, 1e-4, 1e-12, 1e-5, 1e-12, 1e-3, 1e-4, 1e-12]
else:  # T-example
    leak_complete = list(range(1, 10))
    matdata = loadmat(
        f"network_data/{network}/test_data_complete_leak{leak_complete[0]}_{network}.mat"
    )
    n_leaks = 9
    n_times = 288
    noise_cov = [1e-12, 1e-4, 1e-12, 1e-5, 1e-12, 1e-3, 1e-4, 1e-12]
N = int(matdata["N"])

norm_loc_array = np.empty([n_leaks, N])
head_rmse_array = np.empty([n_leaks, n_times])
time_est_array = np.empty([n_leaks, 1])
time_loc_array = np.empty([n_leaks, 1])

for leak_ID in leak_complete:

    print(f"############# Processing: leak {leak_ID + 1} #############")

    leak = leak_ID
    if network == "toy_example":
        matdata = loadmat(
            f"network_data/{network}/test_data_complete_leak{leak_ID}_{network}.mat"
        )
    elif network == "LTOWN":
        matdata = loadmat(
            f"network_data/{network}/test_data_{leak_ID}_leak_{network}_delpump.mat"
        )
        leak = int(matdata["l"]) - 1

    wfgo = WaterFactorGraphEstimator(
        network,
        matdata,
        noise_cov,
        leak_ID=leak_ID,
        leak_pipe=leak,
        localization_factor="pressure",
        n_times=n_times,
    )  # , interpolation_method="AW-GSI")

    wfgo.build_estimate_fgo()
    wfgo.estimate()
    wfgo.build_localization_fgo()
    wfgo.localization()

    head_rmse = wfgo.head_err
    head_rmse_array[leak_ID, :] = head_rmse.T
    time_est_array[leak_ID] = wfgo.time_est
    time_loc_array[leak_ID] = wfgo.time_loc
    nl = wfgo.norm_loc

savemat(
    f"results_FGO_{network}.mat",
    {
        "head_rmse_array": head_rmse_array,
        "time_est_array": time_est_array,
        "time_loc_array": time_loc_array,
        "norm_loc_array": norm_loc_array,
    },
)
