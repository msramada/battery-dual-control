function simulate_CS(x₀₀, Σ₀₀, T, L; u_noise_cov = 0.001)
    x_true = x₀₀ + sqrt(Σ₀₀) * randn(2)
    x_true = clamp.(x_true, 0.0, 1.0)
    n = size(A, 1)
    candidate_Us = [zeros(2, N) for _ in 1:L]
    X_rec = [x₀₀ for _ in 1:T]
    U_rec = [x₀₀ for _ in 1:T]
    Σ_rec = [Σ₀₀ for _ in 1:T]
    X_true_rec = [x_true for _ in 1:T]
    for k in 1:T
        x₀₀ = clamp.(x₀₀, 0.0, 1.0)
        X_rec[k] = x₀₀
        Σ_rec[k] = Σ₀₀
        X_true_rec[k] = x_true
        x_candidates = [clamp.(x₀₀ + sqrt(Σ₀₀) * randn(n,), 0.0, 1.0) for _ in 1:L]
        candidate_Us[1] = apply_MPC(x₀₀, A, B, N, Q * [1 0; 0 1], R * [1 0; 0 1])[1]
        # Threads.@threads 
        for j in 2:L
            candidate_Us[j] = apply_MPC(x_candidates[j], A, B, N, Q * [1 0; 0 1], R * [1 0; 0 1])[1] + sqrt(u_noise_cov) .* randn(2, N)
            #sqrt(tr(Σ₀₀)) .* randn(2, N)
            #candidate_Us[j] = candidate_Us[1] + sqrt(Σ₀₀) * randn(2, N)
        end
        u = trajectory_pick(CS, candidate_Us, x₀₀, Σ₀₀)
        U_rec[k] = u
        x_true = state_dynamics(x_true, u) + sqrt(W) * randn(n)
        x_true = clamp.(x_true, 0.0, 1.0)
        y = measurement_dynamics(x_true) + sqrt(V) * randn(n)
        x₀₀, Σ₀₀ = update(x₀₀, Σ₀₀, u, y, CS.eKF, mode = "measurement")
    end
    return X_rec, U_rec, Σ_rec, X_true_rec
end

function simulate_mpc(x₀₀, Σ₀₀, T)
    x_true = x₀₀ + sqrt(Σ₀₀) * randn(2)
    x_true = clamp.(x_true, 0.0, 1.0)
    n = size(A, 1)

    X_rec_mpc = [x₀₀ for _ in 1:T]
    U_rec_mpc = [x₀₀ for _ in 1:T]
    Σ_rec_mpc = [Σ₀₀ for _ in 1:T]
    X_true_rec_mpc = [x_true for _ in 1:T]
    for k in 1:T
        X_rec_mpc[k] = x₀₀
        Σ_rec_mpc[k] = Σ₀₀
        X_true_rec_mpc[k] = x_true
        U = apply_MPC(x₀₀, A, B, N, Q * [1 0; 0 1], R * [1 0; 0 1])[1]
        u = U[:,1]
        U_rec_mpc[k] = u
        x_true = state_dynamics(x_true, u) + sqrt(W) * randn(2)
        x_true = clamp.(x_true, 0.0, 1.0)
        y = measurement_dynamics(x_true) + sqrt(V) * randn(2)
        x₀₀, Σ₀₀ = update(x₀₀, Σ₀₀, u, y, CS.eKF, mode = "measurement")
        x₀₀ = clamp.(x₀₀, 0.0, 1.0)
    end
    return X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc
end

