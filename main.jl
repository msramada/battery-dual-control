
using Revise
using Plots, LinearAlgebra
using JLD2
using JuMP, Ipopt, MadNLP

include("./src/eKF.jl")
include("./src/MPCs.jl")
include("./simulate.jl")

n = 3  # state dimension
m = 3 # control dimension
d = 3 # output dimension

η = 1 # Efficiency
Q_nom = 2.2 # Nominal capacity
dt = 1.0 # Discretization time
A = I(n)
B = dt * η / Q_nom * I(m)

function state_dynamics(SOC, I)
    SOC = A * SOC + B * I
    return SOC
end

function measurement_dynamics(SOC)

    # Li-ion from Wang et al. 2021,
    "Lithium-Ion Battery SOC Estimation Based on Adaptive Forgetting Factor Least Squares Online Identification and Unscented Kalman Filter"
    OCV_Li = -43.1 * SOC[1]^6 + 155.4 * SOC[1]^5 - 215.7 * SOC[1]^4 + 146.6 * SOC[1]^3 - 50.16 * SOC[1]^2 + 8.674 * SOC[1]

    # From Stroe et al. 2018
    "Influence of Battery Parametric Uncertainties on the State-of-Charge Estimation of Lithium Titanate Oxide-Based Batteries"
    OCV_LTO = 2.1737*SOC[2] - 13.65 * SOC[2]^2 + 31.77 * SOC[2]^3 + 39.06*SOC[2]^4 - 329.2*SOC[2]^5 + 632.1*SOC[2]^6 - 526.2*SOC[2]^7 + 164.8*SOC[2]^8 

    # From Propp et al. 2017
    "Kalman-variant estimators for state of charge in lithium-sulfur batteries"
    OCV_LiS = 0.48*SOC[3] -8.36*SOC[3]^2 + 72.94*SOC[3]^3 - 364.76*SOC[3]^4 + 1107.76*SOC[3]^5 - 2066.02*SOC[3]^6 + 2291.23*SOC[3]^7 - 1372.71*SOC[3]^8 + 339.78*SOC[3]^9

   return [OCV_Li; OCV_LTO; OCV_LiS]
end


W = 0.1 * I(n) # process noise covariance
V = 0.1 * I(d); # measurement noise covariance



eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)

N = 8 # prediction horizon length
Q = 1.0
R = 0.1
set_point = 1.0


function running_cost(x, u)
    cost = Q * (sum(x) - set_point) ^ 2  + u' * R * u
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += (Q * (x[i] - x[j])^2)^2
        end
    end

    return cost
end

function running_cost_stochastic(info_state, u)
    x = info_state[1:n]
    Σ = info_state[n+1:end]
    Σ = reshape(Σ, n, n)
    cost = running_cost(x, u) + Q * tr(Σ)
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += Q * (Σ[i, i] - 2 * Σ[i, j] + Σ[j, j])
        end
    end
    return cost
end



function constraint_function(x, u)
    u_max = 5.0
    return [-x; x .- 1; u .- u_max; -u .- u_max]
end

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
    Σ₀₀ = info_state[n+1:end]
    Σ₀₀ = reshape(Σ₀₀, n, n)
    x₁₁, Σ₁₁ = update_predict(x₀₀, Σ₀₀, u₀, eKF)
    Σ₁₁ = vec(Σ₁₁)
    return [x₁₁; Σ₁₁]
end

function constraint_function_stochastic(x, u)
    u_max = 5.0
    x = x[1:n]
    return [-x; x .- 1; u .- u_max; -u .- u_max]
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

x₀₀ = ones(n)*0.1
Σ₀₀ = 0.5 .* Matrix{Float64}(I, n, n)
L = 2 # number of candidate trajectories
num_simulations = 100
T = 50



function simulation_run()
    X_rec, U_rec, Σ_rec, X_true_rec = simulate_nonlinear_MPC(nonlinear_problem, linear_problem,
                                                        x₀₀, Σ₀₀, T, L; u_noise_cov = 0.01)
    achieved_cost = sum([running_cost(X_true_rec[k], U_rec[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec
end


function run_nonlinear_mpcs()
    cost_rec = zeros(num_simulations)
    est_err_rec = zeros(num_simulations)
    x_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
    u_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
    x_true_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations)  
    cov_rec = Vector{Vector{Matrix{Float64}}}(undef, num_simulations) 


    for i in 1:num_simulations
        println("Simulation: ", i)

        current_time = time()
        achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec = simulation_run()
        println(time()-current_time)

        cost_rec[i] = achieved_cost
        est_err_rec[i] = achieved_est_err
        x_rec[i] = X_rec
        u_rec[i] = U_rec
        x_true_rec[i] = X_true_rec
        cov_rec[i] = Σ_rec
    end
    println("Average Achieved Cost: ", sum(cost_rec) / num_simulations)
    println("Average Achieved Estimation Error: ", sum(est_err_rec) / num_simulations)

    return cost_rec, est_err_rec, x_rec, u_rec, x_true_rec, cov_rec
end



function simulate_run_mpc()
    X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_mpc(linear_problem, x₀₀, Σ₀₀, T)
    achieved_cost = sum([running_cost(X_true_rec_mpc[k], U_rec_mpc[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc 
end


function run_linear_mpcs()

    cost_rec_mpc = zeros(num_simulations)
    est_err_rec_mpc = zeros(num_simulations)
    x_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
    u_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
    x_true_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations)  
    cov_rec_mpc = Vector{Vector{Matrix{Float64}}}(undef, num_simulations)

    for i in 1:num_simulations
        current_time = time()
        achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_run_mpc()
        println(time()-current_time)
        cost_rec_mpc[i] = achieved_cost
        est_err_rec_mpc[i] = achieved_est_err
        x_rec_mpc[i] = X_rec_mpc
        u_rec_mpc[i] = U_rec_mpc
        x_true_rec_mpc[i] = X_true_rec_mpc
        cov_rec_mpc[i] = Σ_rec_mpc
    end
    println("Average Achieved Cost: ", sum(cost_rec_mpc) / num_simulations)
    println("Average Achieved Estimation Error: ", sum(est_err_rec_mpc) / num_simulations)
    
    return cost_rec_mpc, est_err_rec_mpc, x_rec_mpc, u_rec_mpc, x_true_rec_mpc, cov_rec_mpc
end

RUN_SIMS = false

if RUN_SIMS
    println("Running nonlinear MPC simulations...")
    cost_rec, est_err_rec, x_rec, u_rec, x_true_rec, cov_rec = run_nonlinear_mpcs()
    println("Running linear MPC simulations...")
    cost_rec_mpc, est_err_rec_mpc, x_rec_mpc, u_rec_mpc, x_true_rec_mpc, cov_rec_mpc = run_linear_mpcs()

    println("Average Achieved Cost Change: % ", (sum(cost_rec) - sum(cost_rec_mpc)) / sum(cost_rec_mpc) * 100)
    println("Average Achieved Estimation Error Change: % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc) * 100)

    SAVE_DATA = true
    if SAVE_DATA
        @save "simulation_results.jld2" x_rec u_rec cov_rec x_true_rec cost_rec est_err_rec x_rec_mpc u_rec_mpc cov_rec_mpc x_true_rec_mpc cost_rec_mpc est_err_rec_mpc
    end
end


