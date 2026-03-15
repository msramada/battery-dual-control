battery_chemistries = ["Li-ion", "Li-ion", "Li-ion", "LTO", 
        "LTO", "LTO", "Li-S", "Li-S", "Li-S"] # battery chemistries for each battery

n = length(battery_chemistries)  # state dimension (number of batteries)
m = n # control dimension
d = n # output dimension

η = 1 # Efficiency
Q_nom = 2.2 # Nominal capacity
dt = 1.0 # Discretization time
A = I(n)
B = dt * η / Q_nom * I(m)

W = 0.02 * I(n) # process noise covariance
V = 0.02 * I(d); # measurement noise covariance

N = 8 # prediction horizon length
Q = 1.0
R = 0.1
u_max = 0.5
T = 50 # simulation length

# Obs1 increasing the baseline increases cost as allocation becomes more tight
set_point(k) = 5.0 + 1.0 * sin(2 * pi / (1 * T) * k) # time-varying set point for SOC

num_simulations = 50 # number of simulations to run for each MPC

# Initial state and covariance
x₀₀ = ones(n) * 0.1
Σ₀₀ = 0.1 .* Matrix{Float64}(I, n, n)

function state_dynamics(SOC, I)
    SOC = A * SOC + B * I
    return SOC
end

function battery_dynamic_per_chemistry(SOC, chemistry)
    if chemistry == "Li-ion"
        # Li-ion from Wang et al. 2021,
        "Lithium-Ion Battery SOC Estimation Based on Adaptive Forgetting Factor Least Squares Online Identification and Unscented Kalman Filter"
        OCV = -43.1 * SOC^6 + 155.4 * SOC^5 - 215.7 * SOC^4 + 146.6 * SOC^3 - 50.16 * SOC^2 + 8.674 * SOC
    elseif chemistry == "LTO"
        # From Stroe et al. 2018
        "Influence of Battery Parametric Uncertainties on the State-of-Charge Estimation of Lithium Titanate Oxide-Based Batteries"
        OCV = 2.1737*SOC - 13.65 * SOC^2 + 31.77 * SOC^3 + 39.06*SOC^4 - 329.2*SOC^5 + 632.1*SOC^6 - 526.2*SOC^7 + 164.8*SOC^8 
    elseif chemistry == "Li-S"
        # From Propp et al. 2017
        "Kalman-variant estimators for state of charge in lithium-sulfur batteries"
        OCV = 0.48*SOC -8.36*SOC^2 + 72.94*SOC^3 - 364.76*SOC^4 + 1107.76*SOC^5 - 2066.02*SOC^6 + 2291.23*SOC^7 - 1372.71*SOC^8 + 339.78*SOC^9
    else
        error("Unknown chemistry type")
    end
   return OCV
end

function measurement_dynamics(SOC)
    OCV = battery_dynamic_per_chemistry.(SOC, battery_chemistries) 
   return OCV
end


function running_cost(k, x, u)
    cost = Q * (sum(x) - set_point(k)) ^ 2  + 0.0 * u' * R * u
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += 0.2 * Q * (x[i] - x[j])^2
        end
    end
    return cost
end

function running_cost_stochastic(k, info_state, u)
    x = info_state[1:n]
    Σ = info_state[n+1:end]
    Σ = reshape(Σ, n, n)
    cost = running_cost(k, x, u) + Q * tr(Σ)
    # penalize the difference between SOCs
    for i in 1:n
        for j in i+1:n
            cost += Q * (Σ[i, i] - 2 * Σ[i, j] + Σ[j, j])
        end
    end
    return cost
end

function constraint_function(x, u)
    x = x[1:n]
    return [-x; x .- 1; u .- u_max; -u .- u_max]
end
