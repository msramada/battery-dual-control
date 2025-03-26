
mutable struct ExtendedKalmanFilter
    f::Function #f(x,u)
    h::Function #h(x)
    W::Matrix{Float64}
    V::Matrix{Float64}
    function ExtendedKalmanFilter(f, h, W, V)
        if !isposdef(W)
            throw(ArgumentError("W must be a positive semi-definite matrix."))
        end
        if !isposdef(V)
            throw(ArgumentError("V must be a positive semi-definite matrix."))
        end
        new(f, h, W, V)
    end
end


function ∇f(x₀::Vector{Float64}, u₀::Vector{Float64}, eKF::ExtendedKalmanFilter)
	Zygote.jacobian(x -> eKF.f(x, u₀), x₀)[1]
end

function ∇h(x₀::Vector{Float64}, eKF::ExtendedKalmanFilter)
	Zygote.jacobian(x -> eKF.h(x), x₀)[1]
end

function time_update(x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64}, u₀::Vector{Float64}, 
																eKF::ExtendedKalmanFilter)
	x₁₀ = eKF.f(x₀₀, u₀)
	F = ∇f(x₀₀, u₀, eKF) # jacobian of state dynamics
	Σ₁₀ = F * Σ₀₀ * F' + eKF.W
	return x₁₀, Σ₁₀
end

function measurement_update(x₁₀::Vector{Float64}, Σ₁₀::Matrix{Float64}, 
							y₁::Vector{Float64}, eKF::ExtendedKalmanFilter)
	H = ∇h(x₁₀, eKF) # jacobian of measurement dynamics
	L = Σ₁₀ * H' * inv(H * Σ₁₀ * H' + eKF.V + 0.0001*I) #Kalman Gain
	x₁₁ = x₁₀ + L * (y₁ - eKF.h(x₁₀))
	Σ₁₁ = (I - L * H) * Σ₁₀
	return x₁₁, Σ₁₁
end

function update(x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64}, 
			u₀::Vector{Float64}, y₁::Vector{Float64}, eKF::ExtendedKalmanFilter; mode::String = "measurement")
	x₁₀, Σ₁₀ = time_update(x₀₀, Σ₀₀, u₀, eKF)
	if mode == "predict"
		y₁ = eKF.h(x₁₀)
	end
	x₁₁, Σ₁₁ = measurement_update(x₁₀, Σ₁₀, y₁, eKF)
	return x₁₁, Σ₁₁
end
