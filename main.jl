using Revise
using Plots, LinearAlgebra

# calculate LQR cost:
η = 1 # Efficiency
Q_nom = 2.2 # Nominal capacity
dt = 1.0 # Discretization time
A = [1.0 0.0; 0.0 1.0]
B = dt * η / Q_nom * [1.0 0; 0 1.0]

function state_dynamics(SOC, I)
    SOC = A * SOC .+ B * I
    #SOC = clamp.(SOC, 0.0, 1.0)
    return SOC
end

function measurement_dynamics(SOC)
    # LTO
    OCV_LTO = 2.5 + 0.3 * SOC[1] + 0.1 * tanh(8 * (SOC[1] - 0.5)) + 0.05 * sin(5 * π * SOC[1])   
    # LCO
    OCV_LCO = 3.7 + 0.5 * SOC[2] + 0.3 * sin(2 * π * SOC[2])
    return [OCV_LTO; OCV_LCO]
end

function measurement_dynamics_jacobian(SOC)
    d_OCV_LTO = [0.3 + 0.8 * sech(8 * (SOC[1] - 0.5))^2 + 0.25 * π * cos(5 * π * SOC[1]); 0.0]
    d_OCV_LCO = [0.0; 0.5 + 0.6 * π * cos(2 * π * SOC[2])]
    return hcat(d_OCV_LTO, d_OCV_LCO)
end

W = 0.1 * [1.0 0.0; 0.0 1.0]
V = 0.1 * [1.0 0.0; 0.0 1.0];

include("./src/eKF.jl")

eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)

N = 8 # prediction horizon length
Q = 1.0
R = 0.1
set_point = 0.7
running_cost = (x, u) -> Q * (sum(x) - set_point)^2  + u' * R * u
running_cost_stochastic = (x, cov, u) -> Q * (sum(x) - set_point)^2 + Q * tr(cov) + u' * R * u

include("./src/MPCs.jl")
include("./simulate.jl")

function constraint_function(x, u)
    u_max = 5.0
    return [-x; x .- 1; u .- u_max; -u .- u_max]
end

# Define the linear problem
linear_problem = MPC_Prob(
    state_dynamics,
    2, # state dimension
    2, # control dimension
    2, # output dimension
    N, # prediction horizon
    running_cost,
    constraint_function
)

# information state typically refers to the density or anything that represents the state
# and its uncertainty
function time_update_func(x₀₀, Σ₀₀, u₀)
    x₁₀, Σ₁₀ = time_update_predict(x₀₀, Σ₀₀, u₀, eKF)
    return x₁₀, Σ₁₀
end

del_h = x -> ∇h(x, eKF)

nonlinear_problem = MPC_Prob(
    time_update_func, #measurement update will be done within JuMP to avoid the matrix inversion
    2, # state dimension
    2, # control dimension
    2, # output dimension
    N, # prediction horizon
    running_cost_stochastic,
    constraint_function
)
x₀₀ = [0.2; 0.2]
Σ₀₀ = 0.1 * Matrix{Float64}(I, 2, 2)
L = 10
num_simulations = 10
cost_rec = zeros(num_simulations)
est_err_rec = zeros(num_simulations)
T = 50
function simulation_run()
    X_rec, U_rec, _, X_true_rec = simulate_nonlinear_MPC(nonlinear_problem, linear_problem,
                                                        x₀₀, Σ₀₀, T, L; u_noise_cov = 0.01)
    achieved_cost = sum([running_cost(X_true_rec[k], U_rec[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err
end

for i in 1:num_simulations
    println("Simulation: ", i)
    if i==num_simulations
        @time begin
        achieved_cost, achieved_est_err = simulation_run()
        end
    else
        achieved_cost, achieved_est_err = simulation_run()
    end
    cost_rec[i] = achieved_cost
    est_err_rec[i] = achieved_est_err
end
println("Average Achieved Cost: ", sum(cost_rec) / num_simulations)
println("Average Achieved Estimation Error: ", sum(est_err_rec) / num_simulations)

cost_rec_mpc = zeros(num_simulations)
est_err_rec_mpc = zeros(num_simulations)
function simulate_run_mpc()
    X_rec_mpc, U_rec_mpc, _, X_true_rec_mpc = simulate_mpc(linear_problem, x₀₀, Σ₀₀, T)
    achieved_cost = sum([running_cost(X_true_rec_mpc[k], U_rec_mpc[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err 
end
for i in 1:num_simulations
    achieved_cost, achieved_est_err = simulate_run_mpc()
    cost_rec_mpc[i] = achieved_cost
    est_err_rec_mpc[i] = achieved_est_err
end
println("Average Achieved Cost: ", sum(cost_rec_mpc) / num_simulations)
println("Average Achieved Estimation Error: ", sum(est_err_rec_mpc) / num_simulations)

println("Average Achieved Cost Change: % ", (sum(cost_rec)-sum(cost_rec_mpc)) / sum(cost_rec_mpc) * 100)
println("Average Achieved Estimation Error Change: % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc) * 100)