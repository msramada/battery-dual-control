using JuMP, Clarabel

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

    model = Model(Clarabel.Optimizer)
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
        lin_prob.constraint_function(x[:,t+1], u[:,t]) .<= 0.0)
    
    # Find optimal solution
    optimize!(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    if !feasibility
        println("Linear MPC returns infeasible solution")
    end
    return optimal_u
end

function nonlinear_mpc(prob::MPC_Prob, x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64})
    Σ₀₀_vec = vec(Σ₀₀)
    info_state0 = [x₀₀; Σ₀₀_vec]
    A_info = ForwardDiff.jacobian(x -> prob.f(x, zeros(prob.m)), info_state0)
    B_info = ForwardDiff.jacobian(u -> prob.f(info_state0, u), zeros(prob.m))

    # Define the optimization problem
    model = Model(Clarabel.Optimizer)
    #set_optimizer_attribute(model, "tol_feas", 1e-6)
    #model = Model(MadNLP.Optimizer)
    #set_optimizer_attribute(model, "max_iter", 50)
    #set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
    set_silent(model)
    @variable(model, x[i = 1:prob.n+prob.n^2, t = 1:prob.N])
    @variable(model, u[i = 1:prob.m, t = 1:prob.N-1])
    objective = sum(prob.runningcost(x[:,t], u[:,t]) for t in 1:prob.N-1) 
                    + prob.runningcost(x[:,prob.N], 0)
    @objective(model, Min, objective)

    @constraint(model, x[:,1] .== info_state0)
    # New constraint to ensure positive definiteness of the covariance matrix
    @constraint(model, Symmetric(reshape(x[prob.n+1:prob.n+prob.n^2, 1], prob.n, prob.n)) -
        1e-6 * Matrix{Float64}(I, prob.n, prob.n) in PSDCone())
    #@constraint(model, 
    #    x[prob.n+1:prob.n+prob.n^2, 1] .>= 0.0)
    @constraint(model, 
        [t in 1:prob.N-1], 
        x[:,t+1] .== A_info * x[:,t] .+ B_info * u[:,t])

    @constraint(model, 
        [t in 1:prob.N-1], 
        prob.constraint_function(x[:,t+1], u[:,t]) .<= 0.0)

    #obj = @expression(model, objective)
    # Find optimal solution
    optimize!(model)
    optimal_u = value.(u)
    feasibility = (primal_status(model) == FEASIBLE_POINT)
    
    return optimal_u[:,1], feasibility
end