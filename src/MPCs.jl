using JuMP, Ipopt

mutable struct MPC_Prob
    f::Function # state dynamics
    n::Int # state dimension
    m::Int # control dimension
    N::Int # prediction horizon
    runningcost::Function # running cost
    constraint_function::Function
end


function linear_mpc(x₀, lin_prob::MPC_Prob)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:lin_prob.n, 1:lin_prob.N])
    @variable(model, u[i = 1:lin_prob.m, j = 1:lin_prob.N-1])
    @objective(model, Min, 
    sum(lin_prob.runningcost(x[:,t], u[:,t]) for t in 1:T-1)
    + prob.lin_prob(x[:,N], 0))

    # Define the dynamics constraints
    @constraint(model, x[:,1] .== x₀)
    @constraint(model, 
        [t in 1:lin_prob.N-1], 
        x[:,t+1] .== lin_prob.f(x[:,t], u[:,t]))
    @constraint(model, 
        [t in 1:lin_prob.N-1], 
        lin_prob.constraint_function(x[:,t], u[:,t]) .<= 0.0)
    @constraint(model,
    lin_prob.constraint_function(x[:,N], 0) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    obj = objective_value(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u, obj, feasibility
end

function nonlinear_mpc(initial_state::Tuple{Vector{Float64}, Matrix{Float64}},
                       prob::MPC_Prob, initial_u_guess::Matrix{Float64})
    x₀, Σ₀ = initial_state
    # Define the optimization problem
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:prob.n, 1:prob.N])
    @variable(model, Σ[1:prob.n, 1:prob.n, 1:prob.N])
    @variable(model, u[i = 1:prob.m, j = 1:prob.N-1], 
                    start = initial_u_guess[i, j]);
    @objective(model, Min, 
    sum(prob.runningcost(x[:,t], Σ[:,:,t], u[:,t]) for t in 1:T-1)
    + prob.runningcost(x[:,N], Σ[:,:,N], 0))

    # Define the dynamics constraints
    @constraint(model, x[:,1] .== x₀);
    @constraint(model, Σ[:,:,1] .== Σ₀);
    @constraint(model, 
        [t in 1:prob.N-1], 
        x[:,t+1], Σ[:,:,t+1] .== prob.f(x[:,t], Σ[:,:,t], u[:,t]))
    @constraint(model, 
        [t in 1:prob.N-1], 
        prob.constraint_function(x[:,t], u[:,t]) .<= 0.0)
    @constraint(model,
        prob.constraint_function(x[:,N], 0) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    obj = objective_value(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u, obj, feasibility
end