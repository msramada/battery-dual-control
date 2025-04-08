using Revise
using Plots, LinearAlgebra
#import Pkg; Pkg.add("MadNLP")
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


W = 0.1 * [1.0 0.0; 0.0 1.0]
V = 0.1 * [1.0 0.0; 0.0 1.0];

include("./src/eKF.jl")

eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)

N = 8 # prediction horizon length
Q = 1.0
R = 0.1
set_point = 0.7
running_cost = (x, u) -> Q * (sum(x) - set_point)^2  + u' * R * u
n = 2 # state dimension
function running_cost_stochastic(info_state, u)
    x = info_state[1:n]
    Σ = info_state[n+1:end]
    Σ = reshape(Σ, n, n)
    return Q * (sum(x) - set_point)^2 + Q * tr(Σ) + u' * R * u
end

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
    2, # state dimension
    2, # control dimension
    2, # output dimension
    N, # prediction horizon
    running_cost_stochastic,
    constraint_function
)
x₀₀ = [0.2; 0.2]
Σ₀₀ = 0.1 * Matrix{Float64}(I, 2, 2)
L = 10 # number of candidate trajectories
num_simulations = 20
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