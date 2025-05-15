function simulate_nonlinear_MPC(nonlinear_problem, linear_problem, x₀₀, Σ₀₀, T, L; u_noise_cov = 0.001)
    N = linear_problem.N
    n = linear_problem.n
    m = linear_problem.m
    x_true = x₀₀ + sqrt(Σ₀₀) * randn(n)
    x_true = clamp.(x_true, 0.0, 1.0)
    candidate_Us = [zeros(m, N-1) for _ in 1:L]
    X_rec = [x₀₀ for _ in 1:T]
    U_rec = [x₀₀ for _ in 1:T]
    Σ_rec = [Σ₀₀ for _ in 1:T]
    X_true_rec = [x_true for _ in 1:T]
    for k in 1:T
        x₀₀ = clamp.(x₀₀, 0.0, 1.0)
        X_rec[k] = x₀₀
        Σ_rec[k] = Σ₀₀
        X_true_rec[k] = x_true
        for j in 1:L
            if j == 1
                candidate_Us[1] = linear_mpc(x₀₀, linear_problem)
            else
                x_candidate = x₀₀ + sqrt(Σ₀₀) * randn(n)
                x_candidate = clamp.(x_candidate, 0.0, 1.0)
                candidate_Us[j] = linear_mpc(x_candidate, linear_problem) + sqrt(tr(Σ₀₀)) .* randn(m, N-1)
            end
        end
        u, any_feasibility = trajectory_pick(linear_problem, nonlinear_problem, candidate_Us, x₀₀, Σ₀₀)
        if !any_feasibility
            println("Nonlinear MPC cannot find feasible trajectory")
            u = candidate_Us[1][:, 1]
        end
        U_rec[k] = u
        x_true = state_dynamics(x_true, u) + sqrt(W) * randn(n)
        x_true = clamp.(x_true, 0.0, 1.0)
        y = measurement_dynamics(x_true) + sqrt(V) * randn(n)
        x₀₀, Σ₀₀ = update(x₀₀, Σ₀₀, u, y, eKF, mode = "measurement")
    end
    return X_rec, U_rec, Σ_rec, X_true_rec
end


function trajectory_pick(linear_mpc::MPC_Prob, prob::MPC_Prob, u_trajectories::Vector{Matrix{Float64}}, 
    x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64})
    initial_state = (x₀₀, Σ₀₀)
    L = size(u_trajectories, 1)
    costs = zeros(L,)
    optimal_Us = [zeros(prob.m) for _ in 1:L]
    n = prob.n
    Σ₀₀_vec = vec(Σ₀₀)

    Threads.@threads for i = 1:L
        info_state0 = [x₀₀; Σ₀₀_vec]
        initial_u_guess = u_trajectories[i]
        optimal_u, obj, feasibility = nonlinear_mpc(initial_state, prob, initial_u_guess)
        if feasibility
            for k in 1:prob.N-1
                costs[i] += prob.runningcost(info_state0, optimal_u[:, k])/L
                info_state0 = prob.f(info_state0, optimal_u[:, k])
                # x₁ = info_state0[1:n]
                # costs[i] += linear_mpc.runningcost(x₁, optimal_u[:, k])/L
            end
            costs[i] += linear_mpc.runningcost(info_state0, 0.0)/L
            # costs[i] += prob.runningcost(info_state0, 0.0)/L
            # costs[i] = obj
        else
            costs[i] = Inf
        end
        optimal_Us[i] = optimal_u[:, 1]
    end
    any_feasibility = any(costs .< Inf)
    min_cost_index = argmin(costs)
    optimal_u0 = optimal_Us[min_cost_index]
    return optimal_u0, any_feasibility
end

function simulate_mpc(lin_prob::MPC_Prob, x₀₀, Σ₀₀, T)
    n = lin_prob.n
    x_true = x₀₀ + sqrt(Σ₀₀) * randn(n)
    x_true = clamp.(x_true, 0.0, 1.0)
    
    X_rec_mpc = [x₀₀ for _ in 1:T]
    U_rec_mpc = [x₀₀ for _ in 1:T]
    Σ_rec_mpc = [Σ₀₀ for _ in 1:T]
    X_true_rec_mpc = [x_true for _ in 1:T]
    for k in 1:T
        X_rec_mpc[k] = x₀₀
        Σ_rec_mpc[k] = Σ₀₀
        X_true_rec_mpc[k] = x_true
        U = linear_mpc(x₀₀, lin_prob)
        u = U[:,1]
        U_rec_mpc[k] = u
        x_true = state_dynamics(x_true, u) + sqrt(W) * randn(n)
        x_true = clamp.(x_true, 0.0, 1.0)
        y = measurement_dynamics(x_true) + sqrt(V) * randn(size(V, 1))
        x₀₀, Σ₀₀ = update(x₀₀, Σ₀₀, u, y, eKF, mode = "measurement")
        x₀₀ = clamp.(x₀₀, 0.0, 1.0)
    end
    return X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc
end

