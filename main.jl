using Revise
using Plots, LinearAlgebra
#import Pkg; Pkg.add("Zygote"); Pkg.add("Convex"); Pkg.add("SCS")
using Zygote
println("Number of threads used in this instance: ", Threads.nthreads())
##################################################

include("./src/eKF.jl")
include("./src/control_sampler.jl")

# calculate LQR cost:
η = 1 # Efficiency
Q_nom = 2.2 # Nominal capacity
dt = 1.0 # Discretization time
A = [1.0 0.0; 0.0 1.0]
B = dt * η / Q_nom * [1.0 0; 0 1.0]

function state_dynamics(SOC, I)
    SOC = A * SOC .+ B * I
    SOC = clamp.(SOC, 0.0, 1.0)
    return SOC
end

function measurement_dynamics(SOC)
    # LTO
    OCV_LTO = 2.5 + 0.3 * SOC[1] + 0.1 * tanh(8 * (SOC[1] - 0.5)) + 0.05 * sin(8 * π * SOC[1])   
    # LCO
    OCV_LCO = 3.7 + 0.5 * SOC[2] + 0.3 * sin(2 * π * SOC[2])
    return [OCV_LTO; OCV_LCO]
end

W = 0.01 * [1.0 0.0; 0.0 1.0]
V = 0.01 * [1.0 0.0; 0.0 1.0]
eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)

N = 6 # prediction horizon length
Q = 1.0
R = 0.1
set_point = 0.7
running_cost = (x, cov, u) -> Q * (x[1] + x[2] - set_point)^2 + Q * tr(cov) + R * sum(u.^2)
CS = ControlSampler(eKF, N, running_cost)

include("./src/mpc.jl")
include("./simulate.jl")

x₀₀ = [0.2; 0.2]
Σ₀₀ = 0.1 * Matrix{Float64}(I, 2, 2)
L = 100
num_simulations = 6
cost_rec = zeros(num_simulations)
est_err_rec = zeros(num_simulations)
T = 50
function simulation_run()
    X_rec, U_rec, _, X_true_rec = simulate_CS(x₀₀, Σ₀₀, T, L; u_noise_cov = 0.01)
    achieved_cost = sum([CS.running_cost(X_true_rec[k], 0, U_rec[k]) for k in 1:T]) / T
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


"""
X_rec = reduce(hcat, X_rec)
U_rec = reduce(hcat, U_rec)
X_true_rec = reduce(hcat, X_true_rec)
variances = [[Σ_rec[i][1,1] ;Σ_rec[i][2,2]] for i in 1:T]
variances = reduce(hcat, variances)
"""
cost_rec_mpc = zeros(num_simulations)
est_err_rec_mpc = zeros(num_simulations)
function simulate_run_mpc()
    X_rec_mpc, U_rec_mpc, _, X_true_rec_mpc = simulate_mpc(x₀₀, Σ₀₀, T)
    achieved_cost = sum([CS.running_cost(X_true_rec_mpc[k], 0, U_rec_mpc[k]) for k in 1:T]) / T
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

"""
X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_mpc(x₀₀, Σ₀₀, T)
X_rec_mpc = reduce(hcat, X_rec_mpc)
U_rec_mpc = reduce(hcat, U_rec_mpc)
X_true_rec_mpc = reduce(hcat, X_true_rec_mpc)
variances_mpc = [[Σ_rec_mpc[i][1,1] ;Σ_rec_mpc[i][2,2]] for i in 1:T]
variances_mpc = reduce(hcat, variances_mpc)
"""
##################################################
println("Average Achieved Cost Change: % ", (sum(cost_rec)-sum(cost_rec_mpc)) / sum(cost_rec_mpc)*100)
println("Average Achieved Estimation Error Change: % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc)*100)

##################################################