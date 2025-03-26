function apply_MPC(x0, A, B, N, Q, R)

    # Define the variables
    n = size(A, 1)
    m = size(B, 2)
    x = Variable(n, N+1)
    u = Variable(m, N)


    # Define the dynamics constraints
    cost = 0
    for k in 1:N
        cost += quadform((x[1, k] + x[2, k] - set_point), [Q[1,1];;]; assume_psd=true) + quadform(u[:, k], R; assume_psd=true)
    end
    cost += quadform((x[1, N+1] + x[2, N+1] - set_point), [Q[1,1];;]; assume_psd=true)
    # Define the cost function
    problem = minimize(cost)
    
    problem.constraints += x[:, 1] == x0
    for k in 1:N
        problem.constraints += x[:, k+1] == A * x[:, k] + B * u[:, k]
        problem.constraints += -5.0 <= u[:, k]
        problem.constraints += u[:, k] <= 5.0
        problem.constraints += 0.0 <= x[:, k+1]
        problem.constraints += x[:, k+1] <= 1.0
    end
    solve!(problem, SCS.Optimizer; silent_solver = true)
    return u.value, x.value
end