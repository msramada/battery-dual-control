using Revise
using Plots, LinearAlgebra
using JLD2

include("./src/BatteryDualControl.jl")
include("./example_peak_shaving.jl")

# function to invert the covariance matrix for the EKF update


eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)
ukf = UKF.UnscentedKalmanFilter(state_dynamics, measurement_dynamics, W, V)

# Define the linear problem
linear_problem = MPC_Prob(
    state_dynamics,
    n, # state dimension
    m, # control dimension
    d, # output dimension
    N, # prediction horizon
    running_cost,
    constraint_function
)

function info_f(info_state, u₀)
    x₀₀ = info_state[1:n]
    Σ₀₀_vec = info_state[n+1:end]
    Σ₀₀ = reshape(Σ₀₀_vec, n, n)
    x₁₁, Σ₁₁ = update_predict(x₀₀, Σ₀₀, u₀, eKF)
    Σ₁₁ = vec(Σ₁₁)
    return [x₁₁; Σ₁₁]
end


nonlinear_problem = MPC_Prob(
    info_f, #measurement update will be done within JuMP to avoid the matrix inversion
    n, # state dimension
    m, # control dimension
    d, # output dimension
    N, # prediction horizon
    running_cost_stochastic,
    constraint_function
)

function simulation_run()
    X_rec, U_rec, Σ_rec, X_true_rec, X_UKF_rec, X_MHE_rec = simulate_nonlinear_MPC(nonlinear_problem,
                                                            x₀₀, Σ₀₀, T)
    achieved_cost = sum([running_cost(k, X_true_rec[k], U_rec[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    UKF_est_err = sum([norm(X_UKF_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    MHE_est_err = sum([norm(X_MHE_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec, UKF_est_err, MHE_est_err
end


function run_nonlinear_mpcs()
    cost_rec = zeros(num_simulations)
    est_err_rec = zeros(num_simulations)
    x_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    UKF_est_err_rec = zeros(num_simulations)
    MHE_est_err_rec = zeros(num_simulations)
    u_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    x_true_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    cov_rec = Vector{Vector{Matrix{Float64}}}(undef, num_simulations)

    for i in 1:num_simulations
        println("Simulation: ", i)
        current_time = time()
        achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec, UKF_est_err, MHE_est_err = simulation_run()

        cost_rec[i] = achieved_cost
        est_err_rec[i] = achieved_est_err
        UKF_est_err_rec[i] = UKF_est_err
        MHE_est_err_rec[i] = MHE_est_err
        x_rec[i] = X_rec
        u_rec[i] = U_rec
        x_true_rec[i] = X_true_rec
        cov_rec[i] = Σ_rec
    end
    println("Average Achieved Cost: ", sum(cost_rec) / num_simulations)
    println("Average Achieved Estimation Error (EKF): ", sum(est_err_rec) / num_simulations)
    println("Average UKF Estimation Error: ", sum(UKF_est_err_rec) / num_simulations)
    println("Average MHE Estimation Error: ", sum(MHE_est_err_rec) / num_simulations)
    return cost_rec, est_err_rec, x_rec, u_rec, x_true_rec, cov_rec, UKF_est_err_rec, MHE_est_err_rec
end



function simulate_run_mpc()
    X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc, X_UKF_rec_mpc, X_MHE_rec_mpc = simulate_mpc(linear_problem, x₀₀, Σ₀₀, T)
    achieved_cost = sum([running_cost(k, X_true_rec_mpc[k], U_rec_mpc[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    UKF_est_err = sum([norm(X_UKF_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    MHE_est_err = sum([norm(X_MHE_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc, UKF_est_err, MHE_est_err
end


function run_linear_mpcs()

    cost_rec_mpc = zeros(num_simulations)
    est_err_rec_mpc = zeros(num_simulations)
    UKF_est_err_rec_mpc = zeros(num_simulations)
    MHE_est_err_rec_mpc = zeros(num_simulations)
    x_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    u_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    x_true_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations)
    cov_rec_mpc = Vector{Vector{Matrix{Float64}}}(undef, num_simulations)

    for i in 1:num_simulations
        current_time = time()
        achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc, UKF_est_err, MHE_est_err = simulate_run_mpc()
        cost_rec_mpc[i] = achieved_cost
        est_err_rec_mpc[i] = achieved_est_err
        UKF_est_err_rec_mpc[i] = UKF_est_err
        MHE_est_err_rec_mpc[i] = MHE_est_err
        x_rec_mpc[i] = X_rec_mpc
        u_rec_mpc[i] = U_rec_mpc
        x_true_rec_mpc[i] = X_true_rec_mpc
        cov_rec_mpc[i] = Σ_rec_mpc
    end
    println("Average Achieved Cost: ", sum(cost_rec_mpc) / num_simulations)
    println("Average Achieved Estimation Error (EKF): ", sum(est_err_rec_mpc) / num_simulations)
    println("Average UKF Estimation Error: ", sum(UKF_est_err_rec_mpc) / num_simulations)
    println("Average MHE Estimation Error: ", sum(MHE_est_err_rec_mpc) / num_simulations)

    return cost_rec_mpc, est_err_rec_mpc, x_rec_mpc, u_rec_mpc, x_true_rec_mpc, cov_rec_mpc, UKF_est_err_rec_mpc, MHE_est_err_rec_mpc
end

RUN_SIMS = true

if RUN_SIMS
    println("Running nonlinear MPC simulations...")
    cost_rec, est_err_rec, x_rec, u_rec, x_true_rec, cov_rec, UKF_est_err_rec, MHE_est_err_rec = run_nonlinear_mpcs()
    println("Running linear MPC simulations...")
    cost_rec_mpc, est_err_rec_mpc, x_rec_mpc, u_rec_mpc, x_true_rec_mpc, cov_rec_mpc, UKF_est_err_rec_mpc, MHE_est_err_rec_mpc = run_linear_mpcs()

    println("Average Achieved Cost Change: % ", (sum(cost_rec) - sum(cost_rec_mpc)) / sum(cost_rec_mpc) * 100)
    println("Average Achieved Estimation Error Change (EKF): % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc) * 100)
    println("Average UKF Estimation Error Change: % ", (sum(UKF_est_err_rec) - sum(UKF_est_err_rec_mpc)) / sum(UKF_est_err_rec_mpc) * 100)
    println("Average MHE Estimation Error Change: % ", (sum(MHE_est_err_rec) - sum(MHE_est_err_rec_mpc)) / sum(MHE_est_err_rec_mpc) * 100)
    SAVE_DATA = true
    if SAVE_DATA
        @save "simulation_results100.jld2" cov_rec cost_rec est_err_rec UKF_est_err_rec MHE_est_err_rec cov_rec_mpc cost_rec_mpc est_err_rec_mpc UKF_est_err_rec_mpc MHE_est_err_rec_mpc
    end
end


