mutable struct ControlSampler
    eKF::ExtendedKalmanFilter
    N::Int # prediction horizon length
    running_cost::Function
end

function trajectory_pick(CS::ControlSampler, u_trajectories::Vector{Matrix{Float64}},
                                    x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64})
    L = size(u_trajectories, 1)
    costs = zeros(L,)
    n = size(x₀₀, 1)
    for i = 1:L
        x_t = x₀₀
        cov_t = Σ₀₀
        for t = 1:CS.N
            u = u_trajectories[i][:,t]
            costs[i] += CS.running_cost(x_t, cov_t, u)
            x_t, cov_t = update(x_t, cov_t, u, zeros(n,), CS.eKF, mode="predict")
        end
        costs[i] += CS.running_cost(x_t, cov_t, 0)
    end
    _, min_index = findmin(costs)
    return u_trajectories[min_index][:,1]
end