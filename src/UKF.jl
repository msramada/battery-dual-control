module UKF

using LinearAlgebra

mutable struct UnscentedKalmanFilter
    f::Function          # f(x, u)
    h::Function          # h(x)
    W::Matrix{Float64}   # process noise covariance
    V::Matrix{Float64}   # measurement noise covariance
    α::Float64
    β::Float64
    κ::Float64

    function UnscentedKalmanFilter(f, h, W, V; α=1e-2, β=2.0, κ=0.0)
        new(f, h, W, V, α, β, κ)
    end
end

function sigma_points(x::Vector, Σ::Matrix, ukf::UnscentedKalmanFilter)
    n = length(x)
    λ = ukf.α^2 * (n + ukf.κ) - n
    c = n + λ

    # Small jitter for numerical robustness
    S = cholesky(Symmetric(c * Σ + 1e-9I)).L

    X = Matrix{Float64}(undef, n, 2n + 1)
    X[:, 1] = x

    for i in 1:n
        X[:, i + 1]     = x + S[:, i]
        X[:, i + 1 + n] = x - S[:, i]
    end

    wm = fill(1 / (2c), 2n + 1)
    wc = fill(1 / (2c), 2n + 1)

    wm[1] = λ / c
    wc[1] = λ / c + (1 - ukf.α^2 + ukf.β)

    return X, wm, wc
end

function unscented_mean(X::Matrix, w::Vector)
    n, N = size(X)
    μ = zeros(n)
    for i in 1:N
        μ += w[i] * X[:, i]
    end
    return μ
end

function unscented_covariance(X::Matrix, μ::Vector, w::Vector, noise::Matrix)
    n, N = size(X)
    Σ = zeros(n, n)
    for i in 1:N
        d = X[:, i] - μ
        Σ += w[i] * (d * d')
    end
    return Σ + noise
end

function cross_covariance(X::Matrix, μx::Vector, Y::Matrix, μy::Vector, w::Vector)
    nx, N = size(X)
    ny, _ = size(Y)
    Σxy = zeros(nx, ny)
    for i in 1:N
        dx = X[:, i] - μx
        dy = Y[:, i] - μy
        Σxy += w[i] * (dx * dy')
    end
    return Σxy
end

function time_update(x₀₀::Vector, Σ₀₀::Matrix, u₀::Vector, ukf::UnscentedKalmanFilter)
    X, wm, wc = sigma_points(x₀₀, Σ₀₀, ukf)

    n, N = size(X)
    X̄ = Matrix{Float64}(undef, n, N)

    for i in 1:N
        X̄[:, i] = ukf.f(X[:, i], u₀)
    end

    x₁₀ = unscented_mean(X̄, wm)
    Σ₁₀ = unscented_covariance(X̄, x₁₀, wc, ukf.W)

    return x₁₀, Σ₁₀
end

function measurement_update(
    x₁₀::Vector,
    Σ₁₀::Matrix,
    y₁::Vector,
    ukf::UnscentedKalmanFilter
)
    # Rebuild sigma points around predicted state
    X, wm, wc = sigma_points(x₁₀, Σ₁₀, ukf)

    n, N = size(X)
    ydim = length(ukf.h(x₁₀))
    Y = Matrix{Float64}(undef, ydim, N)

    for i in 1:N
        Y[:, i] = ukf.h(X[:, i])
    end

    ȳ₁ = unscented_mean(Y, wm)
    S  = unscented_covariance(Y, ȳ₁, wc, ukf.V)
    Cxy = cross_covariance(X, x₁₀, Y, ȳ₁, wc)

    K = Cxy * inv(S + 1e-9I)

    x₁₁ = x₁₀ + K * (y₁ - ȳ₁)
    Σ₁₁ = Σ₁₀ - K * S * K'

    return x₁₁, Σ₁₁
end

function update(
    x₀₀::Vector,
    Σ₀₀::Matrix,
    u₀::Vector,
    y₁::Vector,
    ukf::UnscentedKalmanFilter;
)
    x₁₀, Σ₁₀ = time_update(x₀₀, Σ₀₀, u₀, ukf)
    x₁₁, Σ₁₁ = measurement_update(x₁₀, Σ₁₀, y₁, ukf)
    return x₁₁, Σ₁₁
end

end