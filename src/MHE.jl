module MHE

using LinearAlgebra, ForwardDiff

"""
Nonlinear Moving Horizon Estimator with log-barrier constraints solved by BFGS.

The MHE cost is augmented with a log-barrier that enforces x ∈ (0,1)^n:

  min_{X[:,1:M]}
      (x₁ - x̄)' P̄⁻¹ (x₁ - x̄)
    + Σ_t (y_t - h(x_t))' V⁻¹ (y_t - h(x_t))
    + Σ_t (x_{t+1} - f(x_t,u_t))' W⁻¹ (x_{t+1} - f(x_t,u_t))
    − μ  Σ_{t,i} [ log(x_{t,i}) + log(1 − x_{t,i}) ]

Warm-start (receding horizon):  the solution from the previous call is reused.
For the steady-state window [k−M+1 … k]:
  • Columns 1 … M−1  ← prev[:,2 … M]  (shift the previous trajectory)
  • Column  M        ← f(prev[:,M], u_{M−1})  (propagate the last state forward)
For the build-up phase the previous trajectory is kept as-is and only the new
column is appended via forward propagation.
On the very first call (no previous solution) x̄ is propagated through dynamics.

The arrival cost (x̄, P̄) is advanced by one EKF time-update whenever the
oldest window entry is dropped.
"""
mutable struct MovingHorizonEstimator
    f::Function          # f(x, u): state transition
    h::Function          # h(x):    measurement model
    W::Matrix{Float64}   # process noise covariance
    V::Matrix{Float64}   # measurement noise covariance
    M::Int               # window / horizon length
    μ::Float64           # log-barrier weight
    u_buffer::Vector{Vector{Float64}}
    y_buffer::Vector{Vector{Float64}}
    x̄::Vector{Float64}   # arrival-cost mean
    P̄::Matrix{Float64}   # arrival-cost covariance
    prev_solution::Union{Nothing, Matrix{Float64}}  # warm-start from last solve

    function MovingHorizonEstimator(f, h, W, V, M, x₀, P₀; μ::Float64 = 1e-4)
        new(f, h, W, V, M, μ,
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            copy(x₀), copy(P₀),
            nothing)
    end
end

# ── Receding-horizon warm-start ──────────────────────────────────────────────
# Reuses the previous BFGS solution, shifted by one step.
# Falls back to forward-propagating x̄ on the very first call.
function _receding_warmstart(mhe::MovingHorizonEstimator; ε::Float64 = 1e-6)
    n     = length(mhe.x̄)
    M_win = length(mhe.y_buffer)

    if mhe.prev_solution === nothing
        # First call: no previous solution, propagate x̄ through dynamics
        X0 = zeros(n, M_win)
        X0[:, 1] = clamp.(mhe.x̄, ε, 1.0 - ε)
        for t in 1:(M_win - 1)
            X0[:, t+1] = clamp.(mhe.f(X0[:, t], mhe.u_buffer[t]), ε, 1.0 - ε)
        end
        return X0
    end

    prev = mhe.prev_solution   # n × M_win_prev
    X0   = zeros(n, M_win)

    if size(prev, 2) < M_win
        # Build-up phase: window just grew by 1, no shift needed
        X0[:, 1:end-1] = prev
    else
        # Steady state: shift the previous trajectory left by one step
        X0[:, 1:end-1] = prev[:, 2:end]
    end

    # Propagate the last kept state to fill the new rightmost column
    X0[:, end] = clamp.(mhe.f(X0[:, end-1], mhe.u_buffer[end-1]), ε, 1.0 - ε)

    return clamp.(X0, ε, 1.0 - ε)
end

# ── Barrier-augmented MHE objective ─────────────────────────────────────────
function _mhe_barrier_objective(mhe::MovingHorizonEstimator,
                                W_inv, V_inv, P_inv, n, M_win)
    μ = mhe.μ
    function obj(z)
        X    = reshape(z, n, M_win)
        T    = eltype(z)
        cost = zero(T)

        # Arrival cost
        Δx = X[:, 1] - mhe.x̄
        cost += dot(Δx, P_inv * Δx)

        # Measurement residuals
        for t in 1:M_win
            r = mhe.y_buffer[t] - mhe.h(X[:, t])
            cost += dot(r, V_inv * r)
        end

        # Dynamics residuals
        for t in 1:(M_win - 1)
            w = X[:, t+1] - mhe.f(X[:, t], mhe.u_buffer[t])
            cost += dot(w, W_inv * w)
        end

        # Log-barrier: returns +Inf immediately if any state leaves (0,1)
        for t in 1:M_win
            for i in 1:n
                xi = X[i, t]
                (xi ≤ zero(T) || xi ≥ one(T)) && return T(Inf)
                cost -= μ * (log(xi) + log(one(T) - xi))
            end
        end

        return cost
    end
    return obj
end

# ── BFGS with Armijo backtracking ────────────────────────────────────────────
function _bfgs(obj, z0::Vector{Float64};
               max_iter::Int  = 20,
               gtol::Float64  = 1e-6)
    m    = length(z0)
    z    = copy(z0)
    g    = ForwardDiff.gradient(obj, z)

    any(!isfinite, g) && return z

    g_norm  = norm(g)
    h_scale = g_norm > 1e-10 ? 1.0 / g_norm : 1.0
    Hinv    = h_scale * Matrix{Float64}(I, m, m)

    for _ in 1:max_iter
        norm(g) < gtol && break

        p = -Hinv * g

        # Reset to steepest descent if p is not a descent direction
        if dot(g, p) >= 0.0
            Hinv = h_scale * Matrix{Float64}(I, m, m)
            p    = -g
        end

        # Armijo backtracking — the barrier handles feasibility via +Inf
        α   = 1.0
        f0  = obj(z)
        !isfinite(f0) && return z
        c1g = 1e-4 * dot(g, p)   # negative (descent)

        for _ in 1:60
            isfinite(obj(z + α * p)) && obj(z + α * p) ≤ f0 + α * c1g && break
            α *= 0.5
            α < 1e-15 && break
        end

        z_new = z + α * p
        g_new = ForwardDiff.gradient(obj, z_new)

        if any(!isfinite, g_new)
            Hinv = h_scale * Matrix{Float64}(I, m, m)
            continue
        end

        s  = z_new - z
        y  = g_new - g
        sy = dot(s, y)

        if sy > 1e-12 * norm(s) * norm(y)
            ρ    = 1.0 / sy
            Hinv = (I - ρ * s * y') * Hinv * (I - ρ * y * s') + ρ * (s * s')
        end

        z = z_new
        g = g_new
    end

    return z
end

# ── public: one-step estimator update ────────────────────────────────────────
"""
    update(_, _, u, y, mhe) -> (x_est, P̄)

Appends (u, y) to the window, warm-starts from the previous MHE solution
shifted by one step (receding horizon), then minimises the barrier-augmented
MHE cost with BFGS.  The solution is stored for the next call.
"""
function update(::Any, ::Any, u, y, mhe::MovingHorizonEstimator)
    # ── 1. Shift window if at capacity ────────────────────────────────────────
    if length(mhe.y_buffer) >= mhe.M
        u_old  = mhe.u_buffer[1]
        F      = ForwardDiff.jacobian(x -> mhe.f(x, u_old), mhe.x̄)
        mhe.x̄  = clamp.(mhe.f(mhe.x̄, u_old), 0.0, 1.0)
        mhe.P̄  = Symmetric(F * mhe.P̄ * F' + mhe.W + 1e-8I)
        popfirst!(mhe.u_buffer)
        popfirst!(mhe.y_buffer)
    end

    # ── 2. Append new data ────────────────────────────────────────────────────
    push!(mhe.u_buffer, copy(u))
    push!(mhe.y_buffer, copy(y))

    n     = length(mhe.x̄)
    M_win = length(mhe.y_buffer)

    W_inv = inv(Symmetric(mhe.W + 1e-8I))
    V_inv = inv(Symmetric(mhe.V + 1e-8I))
    P_inv = inv(Symmetric(mhe.P̄ + 1e-8I))

    # ── 3. Receding-horizon warm-start ───────────────────────────────────────
    X0 = _receding_warmstart(mhe)

    # ── 4. BFGS on the barrier-augmented MHE cost ────────────────────────────
    obj   = _mhe_barrier_objective(mhe, W_inv, V_inv, P_inv, n, M_win)
    z_opt = _bfgs(obj, vec(X0))

    # ── 5. Store solution for next call ──────────────────────────────────────
    mhe.prev_solution = reshape(z_opt, n, M_win)

    x_est = clamp.(mhe.prev_solution[:, end], 0.0, 1.0)
    return x_est, mhe.P̄
end

end  # module MHE
