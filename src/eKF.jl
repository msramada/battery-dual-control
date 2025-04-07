using ForwardDiff
# replace with ForwardDiff, after confirming with Michel
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


function ∇f(x₀::Vector, u₀::Vector, eKF::ExtendedKalmanFilter)
	#ForwardDiff.jacobian(x -> eKF.f(x, u₀), x₀)
	return A
end

function ∇h(x₀::Vector, eKF::ExtendedKalmanFilter)
	#ForwardDiff.jacobian(x -> eKF.h(x), x₀)
	return measurement_dynamics_jacobian(x₀)
end




function time_update(x₀₀::Vector, Σ₀₀::Matrix, u₀::Vector, eKF::ExtendedKalmanFilter)
	x₁₀ = eKF.f(x₀₀, u₀)
	F = ∇f(x₀₀, u₀, eKF) # jacobian of state dynamics
	Σ₁₀ = F * Σ₀₀ * F' + eKF.W
	return x₁₀, Σ₁₀
end

function measurement_update(x₁₀::Vector, Σ₁₀::Matrix, y₁::Vector, eKF::ExtendedKalmanFilter)
	H = ∇h(x₁₀, eKF) # jacobian of measurement dynamics
	L = Σ₁₀ * H' * inv(H * Σ₁₀ * H' + eKF.V + 0.0001*I) #Kalman Gain
	x₁₁ = x₁₀ + L * (y₁ - eKF.h(x₁₀))
	Σ₁₁ = (I - L * H) * Σ₁₀
	return x₁₁, Σ₁₁
end

function update(x₀₀::Vector, Σ₀₀::Matrix, 
			u₀::Vector, y₁::Vector, eKF::ExtendedKalmanFilter; mode::String = "measurement")
	x₁₀, Σ₁₀ = time_update(x₀₀, Σ₀₀, u₀, eKF)
	if mode == "predict"
		y₁ = eKF.h(x₁₀)
	end
	x₁₁, Σ₁₁ = measurement_update(x₁₀, Σ₁₀, y₁, eKF)
	return x₁₁, Σ₁₁
end

function time_update_predict(x₀₀, Σ₀₀, u₀, eKF::ExtendedKalmanFilter)
	x₁₀ = eKF.f(x₀₀, u₀)
	F = ∇f(x₀₀, u₀, eKF) # jacobian of state dynamics
	Σ₁₀ = F * Σ₀₀ * F' + eKF.W
	return x₁₀, Σ₁₀
end

function measurement_update_predict(x₁₀, Σ₁₀, eKF::ExtendedKalmanFilter)
	H = ∇h(x₁₀, eKF) # jacobian of measurement dynamics
	L = Σ₁₀ * H' / (H * Σ₁₀ * H' + eKF.V) #Kalman Gain
	x₁₁ = x₁₀
	Σ₁₁ = (Matrix{Float64}(I, size(eKF.W)) - L * H) * Σ₁₀
	return x₁₁, Σ₁₁
end