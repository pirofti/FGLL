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

import time
from types import SimpleNamespace

import gtsam
import numpy as np
import wntr
from gtsam import noiseModel, symbol

from water_factors import (compute_B, error_func, error_localization,
                           obtain_ps, rmse)


class WaterFactorGraphEstimator:

    def __init__(
        self,
        network_name,
        mat_data,
        noise_cov,
        n_times=2,
        leak_ID="0",
        leak_pipe=0,
        localization_factor="pressure",
        interpolation_method="GSI",
        solver="iSAM",
        enable_demand=1,
        prior_var=1,
    ):

        # Set these before running the script
        self.localization_factor = localization_factor
        self.leak_ID = leak_ID
        self.solver = solver
        self.interpolation_method = interpolation_method
        self.enable_demand = enable_demand

        # Set variance: smaller is more trust, larger is less trust
        self.prior_var = prior_var
        self.sensorized_var = noise_cov[0]
        self.interpolated_var = noise_cov[1]
        self.transition_var = noise_cov[2]
        self.demand_head_var = noise_cov[3]
        self.measurement_demand_var = noise_cov[4]
        self.constraint_demand_var = noise_cov[5]
        # leak
        self.head_res_var = noise_cov[6]
        self.loc_const_var = noise_cov[7]

        self.N = int(mat_data["N"])
        self.Ss = mat_data["Ss"]
        self.pressure_sensors = mat_data["pressure_sensors"] - 1
        self.pressure_sensors = self.pressure_sensors.squeeze()
        self.demand_sensors = mat_data["demand_sensors"].squeeze() - 1

        if network_name == "toy_example":
            self.head = mat_data["Head"]
            self.head_nom = mat_data["Head_nom"]
            self.demand = mat_data["Demand"]
            self.demand_nom = mat_data["Demand_nom"]
            self.extr_nodes = self.pressure_sensors
            self.extr_nodes_dem = self.demand_sensors
            self.Sd = self.Ss
        elif network_name == "Modena":
            self.head = mat_data["Head"][0, leak_pipe]
            self.head_nom = mat_data["Head_nom"][0, leak_pipe]
            self.demand = mat_data["Demand"][0, leak_pipe]
            self.demand_nom = mat_data["Demand_nom"][0, leak_pipe]
            self.Sd = mat_data["Sd"]
            self.extr_nodes = self.pressure_sensors
            self.extr_nodes_dem = self.demand_sensors
        else:  # LTOWN
            self.head = mat_data["Head"][0, leak_pipe]
            self.head_nom = mat_data["Head_nom"][0, leak_pipe]
            self.demand = mat_data["Demand"][0, leak_pipe]
            self.demand_nom = mat_data["Demand_nom"][0, leak_pipe]
            self.extr_nodes = range(self.head_nom.shape[0])
            self.extr_nodes_dem = self.extr_nodes
            self.Sd = self.Ss

        self.E = mat_data["E"] - 1
        self.T = mat_data["T"]
        self.invT = mat_data["invT"]
        self.n_sensor_readings = n_times
        self.pipe = mat_data["pipe"]
        self.pipe_props = SimpleNamespace()
        self.pipe_props.roughness = mat_data["roughness"]
        self.pipe_props.lengths = mat_data["lengths"]
        self.pipe_props.diameters = mat_data["diameters"]
        self.ps = mat_data["Ps"]

        self.time_est = 0
        self.time_loc = 0

        self.interpolated_nodes = np.setdiff1d(np.arange(self.N), self.pressure_sensors)
        self.S = np.eye(self.N)
        self.Sns = self.S[self.interpolated_nodes, :]
        self.n_interpolated_nodes = len(self.interpolated_nodes)
        self.hw_exp = 1 / 1.852

        # FGO Configuration
        self.prior_noise = noiseModel.Gaussian.Covariance(np.eye(self.N))
        self.transition_noise = noiseModel.Gaussian.Covariance(
            self.transition_var * np.eye(self.N)
        )
        self.head_estimate_nom = np.empty([self.N, self.n_sensor_readings])
        self.demand_estimate_nom = np.empty([self.N, self.n_sensor_readings])
        self.demand_estimate_h_nom = np.empty([self.N, self.n_sensor_readings])

        self.wn = wntr.network.WaterNetworkModel(f"{network_name}.inp")

        # hack to pass extra variables to gtsam error functions
        error_func.counter = 0
        error_func.invT = self.invT
        error_func.E = self.E
        error_func.pipe = self.pipe

        error_localization.counter = 0
        error_localization.invT = self.invT
        error_localization.E = self.E
        error_localization.pipe = self.pipe

        self.factor_graph_nom = gtsam.NonlinearFactorGraph()
        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.norm_loc = {}
        self.head_err = {}

    def build_estimate_fgo(
        self,
    ):
        # Prior
        prior_v_nom = self.ps @ self.head_nom[self.extr_nodes, 0]
        prior_v_nom[self.pressure_sensors] = self.head_nom[self.extr_nodes, 0]
        self.factor_graph_nom.add(
            gtsam.PriorFactorVector(symbol("x", 0), prior_v_nom, self.prior_noise)
        )
        demand_noise = noiseModel.Diagonal.Sigmas(
            self.demand_head_var * np.ones(self.N)
        )

        # sensor nodes have a smaller variance corresponding to a higher trust value
        noise_sigmas = np.ones(self.N)
        noise_sigmas[self.interpolated_nodes] = (
            self.interpolated_var * noise_sigmas[self.interpolated_nodes]
        )
        noise_sigmas[self.pressure_sensors] = (
            self.sensorized_var * noise_sigmas[self.pressure_sensors]
        )
        measurement_noise = noiseModel.Diagonal.Sigmas(
            self.measurement_demand_var * np.ones(len(self.interpolated_nodes))
        )  # noise_sigmas)
        measurement_demand_noise = noiseModel.Diagonal.Sigmas(
            self.measurement_demand_var * np.ones(len(self.demand_sensors))
        )
        constraint_demand_noise = noiseModel.Diagonal.Sigmas(
            self.constraint_demand_var * np.ones(1)
        )
        demand_transition_noise = noiseModel.Gaussian.Covariance(np.diag(noise_sigmas))

        # Add factors for each sensor reading
        for k in range(self.n_sensor_readings - 1):

            # HEAD

            # Prediction
            delta_state_nom = self.ps @ (
                self.head_nom[self.extr_nodes, k + 1]
                - self.head_nom[self.extr_nodes, k]
            )  # np.zeros(self.N)
            delta_state_nom[self.pressure_sensors] = (
                self.head_nom[self.extr_nodes, k + 1]
                - self.head_nom[self.extr_nodes, k]
            )
            self.factor_graph_nom.add(
                gtsam.BetweenFactorVector(
                    symbol("x", k),
                    symbol("x", k + 1),
                    delta_state_nom,
                    self.transition_noise,
                )
            )

            # Pressure measurements
            z_prior_nom = self.ps @ self.head_nom[self.extr_nodes, k]
            z_prior_nom[self.pressure_sensors] = self.head_nom[self.extr_nodes, k]

            z_interpolated_nom = self.ps @ self.head_nom[self.extr_nodes, k + 1]
            z_interpolated_nom[self.pressure_sensors] = self.head_nom[
                self.extr_nodes, k + 1
            ]
            residual_nom = z_interpolated_nom - z_prior_nom

            jacobian_factor_nom = gtsam.JacobianFactor(
                symbol("x", k + 1),
                self.Sns,
                residual_nom[self.interpolated_nodes],
                measurement_noise,
            )

            self.factor_graph_nom.add(gtsam.LinearContainerFactor(jacobian_factor_nom))

            # DEMANDS

            if self.enable_demand == 1:

                # Prediction
                delta_state_demand_nom = np.zeros(self.N)
                delta_state_demand_nom[self.demand_sensors] = (
                    self.demand_nom[self.extr_nodes_dem, k + 1]
                    - self.demand_nom[self.extr_nodes_dem, k]
                )
                self.factor_graph_nom.add(
                    gtsam.BetweenFactorVector(
                        symbol("d", k),
                        symbol("d", k + 1),
                        delta_state_demand_nom,
                        demand_transition_noise,
                    )
                )

                residual_demand_nom = (
                    self.demand_nom[self.extr_nodes_dem, k + 1]
                    - self.demand_nom[self.extr_nodes_dem, k]
                )
                jacobian_demand_factor_nom = gtsam.JacobianFactor(
                    symbol("d", k + 1),
                    self.Sd,
                    residual_demand_nom,
                    measurement_demand_noise,
                )
                self.factor_graph_nom.add(
                    gtsam.LinearContainerFactor(jacobian_demand_factor_nom)
                )

                self.factor_graph_nom.add(
                    gtsam.LinearContainerFactor(
                        gtsam.JacobianFactor(
                            symbol("d", k + 1),
                            np.ones([1, self.N]),
                            np.zeros(1),
                            constraint_demand_noise,
                        )
                    )
                )

                self.factor_graph_nom.add(
                    gtsam.CustomFactor(
                        demand_noise,
                        [gtsam.symbol("x", k + 1), gtsam.symbol("d", k + 1)],
                        error_func,
                    )
                )

        return self.factor_graph_nom

    def estimate(
        self,
    ):
        # Update estimation
        hFGO_v_nom = self.ps @ self.head_nom[self.extr_nodes]
        hFGO_v_nom[self.pressure_sensors, :] = self.head_nom[self.extr_nodes]
        demand_v_nom = np.zeros([self.N, self.n_sensor_readings])
        demand_v_nom[self.demand_sensors, :] = self.demand_nom[
            self.extr_nodes_dem, : self.n_sensor_readings
        ]
        new_values_nom = gtsam.Values()
        for t in range(self.n_sensor_readings):
            new_values_nom.insert(symbol("x", t), hFGO_v_nom[:, t])
            if self.enable_demand == 1:
                new_values_nom.insert(symbol("d", t), demand_v_nom[:, t])

        print("Nominal optimization")
        start_time = time.time()
        if self.solver == "iSAM":
            iSAM = gtsam.ISAM2()
            iSAM.update(self.factor_graph_nom, new_values_nom)
            current_result_nom = iSAM.calculateEstimate()
        elif self.solver == "gauss":
            params = gtsam.GaussNewtonParams()
            optimizer = gtsam.GaussNewtonOptimizer(
                self.factor_graph_nom, new_values_nom, params
            )
            current_result_nom = optimizer.optimize()
        else:
            print(f"Unknown solver {self.solver}!")

        end_time = time.time()

        elapsed = end_time - start_time
        self.time_est = elapsed
        print(f"Elapsed time: {elapsed:.4f} seconds")
        print("done.")

        # Obtain new result
        head_err_nom = np.empty([self.n_sensor_readings])

        for t in range(self.n_sensor_readings):
            self.head_estimate_nom[:, t] = current_result_nom.atVector(symbol("x", t))
            B = compute_B(self.head_estimate_nom[:, t], self.E)
            self.demand_estimate_h_nom[:, t] = (
                -B @ (self.invT @ B.T @ self.head_estimate_nom[:, t]) ** self.hw_exp
            )
            head_err_nom[t] = rmse(self.head_estimate_nom[:, t], self.head_nom[:, t])

        # Final RMSE for this instance
        print("done")
        print(f"Average Head RMSE (nominal): {np.average(head_err_nom)}")

    def build_localization_fgo(
        self,
    ):

        # Prior
        if self.interpolation_method == "AW-GSI":
            ps = obtain_ps(
                self.head_estimate_nom[:, 0],
                self.E,
                self.pipe_props,
                self.Ss,
                self.hw_exp,
            )
        else:
            ps = self.ps
        prior_v = ps @ self.head[self.extr_nodes, 0]
        prior_v[self.pressure_sensors] = self.head[self.extr_nodes, 0]
        self.factor_graph.add(
            gtsam.PriorFactorVector(symbol("x", 0), prior_v, self.prior_noise)
        )

        head_res_noise = noiseModel.Diagonal.Sigmas(self.head_res_var * np.ones(self.N))
        loc_const_noise = noiseModel.Diagonal.Sigmas(
            self.loc_const_var * np.ones(self.N)
        )
        demand_noise = noiseModel.Diagonal.Sigmas(
            self.demand_head_var * np.ones(self.N)
        )
        measurement_noise = noiseModel.Diagonal.Sigmas(
            self.measurement_demand_var * np.ones(len(self.interpolated_nodes))
        )  # noise_sigmas)
        measurement_demand_noise = noiseModel.Diagonal.Sigmas(
            self.measurement_demand_var * np.ones(len(self.demand_sensors))
        )
        constraint_demand_noise = noiseModel.Diagonal.Sigmas(
            self.constraint_demand_var * np.ones(1)
        )

        # Add factors for each sensor reading
        for k in range(self.n_sensor_readings - 1):
            if self.interpolation_method == "AW-GSI":
                ps = obtain_ps(
                    self.head_estimate_nom[:, k],
                    self.E,
                    self.pipe_props,
                    self.Ss,
                    self.hw_exp,
                )
            else:
                ps = self.ps
            # Prediction
            delta_state = ps @ (
                self.head[self.extr_nodes, k + 1] - self.head[self.extr_nodes, k]
            )  # np.zeros(N)
            delta_state[self.pressure_sensors] = (
                self.head[self.extr_nodes, k + 1] - self.head[self.extr_nodes, k]
            )
            self.factor_graph.add(
                gtsam.BetweenFactorVector(
                    symbol("x", k),
                    symbol("x", k + 1),
                    delta_state,
                    self.transition_noise,
                )
            )

            # Pressure measurements
            z_prior = ps @ self.head[self.extr_nodes, k]
            z_interpolated = ps @ self.head[self.extr_nodes, k + 1]
            residual = z_interpolated - z_prior

            jacobian_factor = gtsam.JacobianFactor(
                symbol("x", k + 1),
                self.Sns,
                residual[self.interpolated_nodes],
                measurement_noise,
            )

            self.factor_graph.add(gtsam.LinearContainerFactor(jacobian_factor))

            # DEMANDS

            # Prediction
            delta_state_demand = np.zeros(self.N)
            delta_state_demand[self.demand_sensors] = (
                self.demand[self.extr_nodes_dem, k + 1]
                - self.demand[self.extr_nodes_dem, k]
            )
            self.factor_graph.add(
                gtsam.BetweenFactorVector(
                    symbol("d", k),
                    symbol("d", k + 1),
                    delta_state_demand,
                    self.transition_noise,
                )
            )

            residual_demand = (
                self.demand[self.extr_nodes_dem, k + 1]
                - self.demand[self.extr_nodes_dem, k]
            )
            jacobian_demand_factor = gtsam.JacobianFactor(
                symbol("d", k + 1),
                self.Sd,
                residual_demand,
                measurement_demand_noise,
            )
            self.factor_graph.add(gtsam.LinearContainerFactor(jacobian_demand_factor))

            self.factor_graph.add(
                gtsam.LinearContainerFactor(
                    gtsam.JacobianFactor(
                        symbol("d", k + 1),
                        np.ones([1, self.N]),
                        np.zeros(1),
                        constraint_demand_noise,
                    )
                )
            )

            self.factor_graph.add(
                gtsam.CustomFactor(
                    demand_noise,
                    [gtsam.symbol("x", k + 1), gtsam.symbol("d", k + 1)],
                    error_func,
                )
            )

            # LOCALIZATION FACTOR

            # Localization with demand and pressure residuals
            if self.localization_factor == "demand":
                B = compute_B(self.head_estimate_nom[:, k], self.E)

                # pass time dependent variables
                error_localization.d_n = self.demand_estimate_nom[:, k]
                error_localization.dh_n = (
                    -B @ (self.invT @ B.T @ self.head_estimate_nom[:, k]) ** self.hw_exp
                )

                self.factor_graph.add(
                    gtsam.CustomFactor(
                        head_res_noise,
                        [symbol("x", k + 1), symbol("d", k + 1), symbol("l", k + 1)],
                        error_localization,
                    )
                )
            # Computation of the pressure residual
            elif self.localization_factor == "pressure":
                self.factor_graph.add(
                    gtsam.BetweenFactorVector(
                        symbol("x", k + 1),
                        symbol("l", k + 1),
                        -self.head_estimate_nom[:, k],
                        head_res_noise,
                    )
                )

            # Constraint for the temporal relation between localizations

            self.factor_graph.add(
                gtsam.BetweenFactorVector(
                    symbol("l", k),
                    symbol("l", k + 1),
                    np.zeros(self.N),
                    loc_const_noise,
                )
            )

        return self.factor_graph

    def localization(
        self,
    ):
        # Update estimation
        hFGO_v = self.ps @ self.head[self.extr_nodes]
        hFGO_v[self.pressure_sensors, :] = self.head[self.extr_nodes]
        demand_v = np.zeros([len(self.demand), self.n_sensor_readings])
        demand_v[self.demand_sensors, :] = self.demand[
            self.extr_nodes_dem, : self.n_sensor_readings
        ]
        new_values = gtsam.Values()
        for t in range(self.n_sensor_readings):
            new_values.insert(symbol("x", t), hFGO_v[:, t])
            new_values.insert(symbol("l", t), np.zeros(self.N))
            new_values.insert(symbol("d", t), demand_v[:, t])

        print("Leak optimization")
        start_time = time.time()
        if self.solver == "iSAM":
            iSAM = gtsam.ISAM2()
            iSAM.update(self.factor_graph, new_values)
            current_result = iSAM.calculateEstimate()
        elif self.solver == "gauss":
            params = gtsam.GaussNewtonParams()
            optimizer = gtsam.GaussNewtonOptimizer(
                self.factor_graph, new_values, params
            )
            current_result = optimizer.optimize()
        else:
            print(f"Unknown optimizer {self.solver}!")
        end_time = time.time()
        elapsed = end_time - start_time
        self.time_loc = elapsed
        print(f"Elapsed time: {elapsed:.4f} seconds")
        print("done.")

        # Obtain new result
        head_estimate = np.empty([self.N, self.n_sensor_readings])
        localization_estimate = np.empty([self.N, self.n_sensor_readings])
        head_err = np.empty([self.n_sensor_readings])

        for t in range(self.n_sensor_readings):
            head_estimate[:, t] = current_result.atVector(symbol("x", t))
            # B = compute_B(head_estimate[:, t], self.E)

            localization_estimate[:, t] = -current_result.atVector(symbol("l", t))
            head_err[t] = rmse(head_estimate[:, t], self.head[:, t])

        # Final RMSE for this instance
        print("done")
        print(f"Average Head RMSE: {np.average(head_err)}")

        loc_v = localization_estimate[:, 0]
        self.norm_loc = (loc_v - np.min(loc_v)) / (np.max(loc_v) - np.min(loc_v))
        self.head_err = head_err
