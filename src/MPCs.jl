using JuMP, Ipopt, MadNLP

mutable struct MPC_Prob
    f::Function # state dynamics
    n::Int # state dimension
    m::Int # control dimension
    o::Int # output dimension
    N::Int # prediction horizon
    runningcost::Function # running cost
    constraint_function::Function
end


function linear_mpc(x₀::Vector{Float64}, lin_prob::MPC_Prob)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:lin_prob.n, 1:lin_prob.N])
    @variable(model, u[1:lin_prob.m, 1:lin_prob.N-1])
    @objective(model, Min, 
    sum(lin_prob.runningcost(x[:,t], u[:,t]) for t in 1:lin_prob.N-1)
                                + lin_prob.runningcost(x[:,lin_prob.N], 0))

    # Define the dynamics constraints
    @constraint(model, x[:,1] .== x₀)
    @constraint(model, 
        [t in 1:lin_prob.N-1], 
        x[:,t+1] .== lin_prob.f(x[:,t], u[:,t]))
    @constraint(model, 
        [t in 1:lin_prob.N-1], 
        lin_prob.constraint_function(x[:,t], u[:,t]) .<= 0.0)
    @constraint(model,
        lin_prob.constraint_function(x[:,lin_prob.N], 0) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    if !feasibility
        println("Linear MPC returns infeasible solution")
    end
    return optimal_u
end

function nonlinear_mpc(initial_state::Tuple{Vector{Float64}, Matrix{Float64}},
                       prob::MPC_Prob, initial_u_guess::Matrix{Float64})
    x₀₀, Σ₀₀ = initial_state
    u₀ = initial_u_guess[:, 1]
    Σ₀₀_vec = vec(Σ₀₀)
    γ = 1
    info_state0 = [x₀₀; Σ₀₀_vec]
    info_state_rec = [info_state0 for _ in 1:prob.N]
    A_full = ForwardDiff.jacobian(x -> prob.f(x, u₀), info_state0)
    B_full = ForwardDiff.jacobian(u -> prob.f(info_state0, u), u₀)
    As = [A_full for _ in 1:prob.N]
    Bs = [B_full for _ in 1:prob.N]
    for k in 1:prob.N-1
        uₖ = initial_u_guess[:, k]
        info_state_next = prob.f(info_state_rec[k], uₖ)
        info_state_rec[k+1] = info_state_next
        As[k+1] = ForwardDiff.jacobian(x -> prob.f(x, uₖ), info_state_next)
        Bs[k+1] = ForwardDiff.jacobian(u -> prob.f(info_state_next, u), uₖ)
    end
    # Define the optimization problem
    model = Model(Ipopt.Optimizer)
    #model = Model(MadNLP.Optimizer)
    #set_optimizer_attribute(model, "max_iter", 50)
    #set_optimizer_attribute(model, "constr_viol_tol", 1e-6)

    set_silent(model)
    @variable(model, x[i = 1:prob.n+prob.n^2, j = 1:prob.N], start = info_state_rec[j][i])
    @variable(model, u[i = 1:prob.m, j = 1:prob.N-1], start = initial_u_guess[i, j])
    objective = sum(prob.runningcost(x[:,t], u[:,t]) for t in 1:prob.N-1) + prob.runningcost(x[:,prob.N], 0)
    regularization = sum((u[i,t] - initial_u_guess[i, t])^2 for t in 1:prob.N-1 for i in 1:prob.m)
    regularization = sum((x[i,t] - info_state_rec[t][i])^2 for t in 1:prob.N for i in 1:prob.n+prob.n^2)
    @objective(model, Min, objective + γ * regularization)

    # Define the dynamics constraints
    @constraint(model, x[:,1] .== info_state0)
    @constraint(model, 
        [t in 1:prob.N-1], 
        x[:,t+1] .== As[t] * x[:,t] + Bs[t] * u[:,t])

    @constraint(model, 
        [t in 1:prob.N-1], 
        prob.constraint_function(x[:,t+1], u[:,t]) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u
end