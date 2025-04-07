using JuMP, Ipopt

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
    obj = objective_value(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u, obj, feasibility
end

function nonlinear_mpc(initial_state::Tuple{Vector{Float64}, Matrix{Float64}},
                       prob::MPC_Prob, initial_u_guess::Matrix{Float64})
    x₀₀, Σ₀₀ = initial_state
    # Define the optimization problem
    model = Model(Ipopt.Optimizer)
    # Register the external function
    #@NLparameter(model, c == 0)  # workaround to initialize NL expressions
    #JuMP.register(model, :f, 3, prob.f; autodiff=true)

    set_silent(model)
    @variable(model, x₁₁[i = 1:prob.n, j = 1:prob.N])
    @variable(model, Σ₁₀[i = 1:prob.n, j = 1:prob.n, k = 2:prob.N])
    @variable(model, Σ₁₁[i = 1:prob.n, j = 1:prob.n, k = 1:prob.N])
    @variable(model, L[i = 1:prob.n, j = 1:prob.o, k = 2:prob.N])
    @variable(model, u[i = 1:prob.m, j = 1:prob.N-1], 
                    start = initial_u_guess[i, j]);
    @objective(model, Min, 
    sum(prob.runningcost(x₁₁[:,t], Σ₁₁[:,:,t], u[:,t]) for t in 1:prob.N-1)
    + prob.runningcost(x₁₁[:,prob.N], Σ₁₁[:,:,prob.N], 0))

    # Define the dynamics constraints
    @constraint(model, x₁₁[:,1] .== x₀₀);
    @constraint(model, Σ₁₁[:,:,1] .== Σ₀₀);

    @constraint(model, 
        [t in 1:prob.N-1], 
        x₁₁[:,t+1] .== prob.f(x₁₁[:,t], Σ₁₁[:,:,t], u[:,t])[1])
    @constraint(model, 
        [t in 1:prob.N-1], 
        Σ₁₀[:,:,t+1] .== prob.f(x₁₁[:,t], Σ₁₁[:,:,t], u[:,t])[2])
    @constraint(model,
    [t in 1:prob.N-1], 
    L[:,:,t+1] * (del_h(x₁₁[:,t+1]) * Σ₁₀[:,:,t+1] * del_h(x₁₁[:,t+1])' + eKF.V) .== Σ₁₀[:,:,t+1] * del_h(x₁₁[:,t+1])')
    @constraint(model, 
    [t in 1:prob.N-1], 
    Σ₁₁[:,:,t+1] .== (1.0 * I(prob.n) - L[:,:,t+1] * del_h(x₁₁[:,t+1])) * Σ₁₀[:,:,t+1] * del_h(x₁₁[:,t+1])')
    @constraint(model, 
        [t in 1:prob.N-1], 
        prob.constraint_function(x₁₁[:,t], u[:,t]) .<= 0.0)
    @constraint(model,
        prob.constraint_function(x₁₁[:,prob.N], 0) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    obj = objective_value(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u, obj, feasibility
end