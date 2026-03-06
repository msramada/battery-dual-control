function simulate_nonlinear_MPC(nonlinear_problem, x₀₀, Σ₀₀, T)
    N = nonlinear_problem.N
    n = nonlinear_problem.n
    m = nonlinear_problem.m
    x_true = x₀₀ + sqrt(Σ₀₀) * randn(n)
    x_true = clamp.(x_true, 0.0, 1.0)
    X_rec = [x₀₀ for _ in 1:T]
    U_rec = [zeros(m) for _ in 1:T]
    Σ_rec = [Σ₀₀ for _ in 1:T]
    X_true_rec = [x_true for _ in 1:T]
    for k in 1:T
        x₀₀ = clamp.(x₀₀, 0.0, 1.0)
        X_rec[k] = x₀₀
        Σ_rec[k] = Σ₀₀
        X_true_rec[k] = x_true
        
        u, feasibile = nonlinear_mpc(nonlinear_problem, x₀₀, Σ₀₀)
        if !feasibile
            println("Nonlinear MPC is infeasible")
        end

        U_rec[k] = u
        x_true = state_dynamics(x_true, u) + sqrt(W) * randn(n)
        x_true = clamp.(x_true, 0.0, 1.0)
        y = measurement_dynamics(x_true) + sqrt(V) * randn(n)
        x₀₀, Σ₀₀ = update(x₀₀, Σ₀₀, u, y, eKF, mode = "measurement")
    end
    return X_rec, U_rec, Σ_rec, X_true_rec
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

