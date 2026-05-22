using Gurobi
using JuMP
using LinearAlgebra
using Random
using Base.Threads
ENV["JULIA_NUM_THREADS"] = 12
# include("plot.jl")
include("util.jl")



println("Num threads: ", nthreads())

using Printf

const EPS = 1e-5
const mt = true




"""
projection_3pm(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)

Computes the 3pm projections of x in the elipsoids (y_i,Q_i,k_i)

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projection
- ϵ::Real:              precision parameter

Returns a p ∈ ℜ^(m,n) matrix with the 3pm projections of x.
"""
function projection_3pm(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env)
	k, _, n = size(Q)
	p = zeros(k, n)

	for i in 1:k
		p[i, :] = polyhedron_projection(x, y[i, :], Q[i, :, :], env)
	end

	return p
end

"""
projection_3pm_mt(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)

Computes the 3pm projections of x in the elipsoids (y_i,Q_i,k_i) in parallel.

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii

Returns a p ∈ ℜ^(m,n) matrix with the 3pm projections of x.
"""
function projection_3pm_mt(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env)
	k, _, n = size(Q)
	p = zeros(k, n)

	Threads.@threads for i in 1:k
		p[i, :] .= polyhedron_projection(x, y[i, :], Q[i, :, :], env)
	end

	return p
end

"""
projection_a3pm(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)

Computes the a3pm projections of x in the elipsoids (y_i,Q_i,k_i)

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii

Returns a p ∈ ℜ^(m,n) matrix with the a3pm projections of x.
"""
function projection_a3pm(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray)

	k, _, n = size(Q)
	p = zeros(eltype(x), k, n)

	for i in 1:k
		aux = Q[i, :, :] * x .- y[i, :] # aux = A_i x - b_i
		g_i, index = findmax(aux)
		if g_i <= 0.0
			p[i, :] .= x
		else
			grad = Q[i, index, :]
			norm_grad = sum(abs.(grad) .^ 2)
			step_size = g_i / norm_grad
			step_size = clamp(step_size, 1e-6, 1.0)
			p[i, :] .= x .- step_size .* grad
		end
	end

	return p
end

"""
projection_a3pm_mt(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)

Computes the a3pm projections of x in the elipsoids (y_i,Q_i,k_i) in parallel

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
Returns a p ∈ ℜ^(m,n) matrix with the a3pm projections of x.
"""
function projection_a3pm_mt(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray)

	k, _, n = size(Q)
	p = zeros(eltype(x), k, n)

	Threads.@threads for i in 1:k
		# sum positive entries of Q[i, :, :]' * x .- y[i, :]
		aux = Q[i, :, :] * x .- y[i, :]
		g_i, index = findmax(aux)
		if g_i <= 0.0
			p[i, :] .= x
		else
			grad = Q[i, index, :]
			norm_grad = sum(abs.(grad) .^ 2)
			step_size = g_i / norm_grad
			step_size = clamp(step_size, 1e-6, 1.0)
			p[i, :] .= x .- step_size .* grad
		end
	end

	return p
end


"""
update_3pm!(x::AbstractVector, p::AbstractMatrix)

Updates x using the 3pm projections p.
-  x::AbstractVector:   size(x) = n, current solution
-  p::AbstractMatrix:   size(p) = (m,n), 3pm projections
-  env::Gurobi.Env:     Gurobi environment to solve update
-  ϵ::Real:             Tolerance parameter
"""
function update_3pm!(x::AbstractVector, p::AbstractMatrix, env::Gurobi.Env, ϵ::Real, iter::Int, xo::AbstractVector)
	m, n = size(p)

	model = direct_model(Gurobi.Optimizer(env))
	set_silent(model)
	set_optimizer_attribute(model, "NumericFocus", 3)

	@variable(model, y_var[1:n])

	@objective(model, Min, 0.5 * sum((y_var[k] - x[k])^2 for k in 1:n))
	cons = []
	for i in 1:m
		push!(cons, @constraint(model, sum((x[k] - p[i, k]) * (y_var[k] - p[i, k]) for k in 1:n) <= 0.0))
	end

	optimize!(model)

	status = termination_status(model)

	if status == MOI.INFEASIBLE
		println("model infeasible")
		write_to_file(model, "model_update.lp")
		error("Infeasible model")
	elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
		# Equivalente ao DualReductions = 0
		println("Infeasible or unbounded. Trying with DualReductions.")
		set_optimizer_attribute(model, "DualReductions", 0)
		optimize!(model)
		status = termination_status(model)
		if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
			println("model infeasible after DualReductions ")
			write_to_file(model, "model_update.lp")
			error("Infeasible model after DualReductions")
		end
	elseif status != MOI.OPTIMAL
		println("Unknown status: ", status)
		write_to_file(model, "model_update.lp")
		error("Model status not optimal")
	end

	x .= value.(y_var)

	return x
end


function update_3pm_halpern!(x::AbstractVector, p::AbstractMatrix, env::Gurobi.Env, ϵ::Real, iter::Int, x0::AbstractVector)
	m, n = size(p)

	model = direct_model(Gurobi.Optimizer(env))
	set_silent(model)
	set_optimizer_attribute(model, "NumericFocus", 3)

	@variable(model, y_var[1:n])

	@objective(model, Min, 0.5 * sum((y_var[k] - x[k])^2 for k in 1:n))
	cons = []
	for i in 1:m
		push!(cons, @constraint(model, sum((x[k] - p[i, k]) * (y_var[k] - p[i, k]) for k in 1:n) <= 0.0))
	end

	optimize!(model)

	status = termination_status(model)

	if status == MOI.INFEASIBLE
		println("model infeasible")
		write_to_file(model, "model_update.lp")
		error("Infeasible model")
	elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
		# Equivalente ao DualReductions = 0
		println("Infeasible or unbounded. Trying with DualReductions.")
		set_optimizer_attribute(model, "DualReductions", 0)
		optimize!(model)
		status = termination_status(model)
		if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
			println("model infeasible after DualReductions ")
			write_to_file(model, "model_update.lp")
			error("Infeasible model after DualReductions")
		end
	elseif status != MOI.OPTIMAL
		println("Unknown status: ", status)
		write_to_file(model, "model_update.lp")
		error("Model status not optimal")
	end
	x .= value.(y_var)
	x .= (1.0 / (iter)) * x0 + (1.0 - (1.0 / (iter))) * x
	return x
end


"""
update_a3pm!(x::AbstractVector, p::AbstractMatrix)

Updates x using the a3pm projections p.
-  x::AbstractVector:   size(x) = n, current solution
-  p::AbstractMatrix:   size(p) = (m,n), 3pm projections
-  ϵ::Real:             precision parameter
"""
function update_a3pm_halpern!(x::AbstractVector, p::AbstractMatrix, iter::Int, x0::AbstractVector)
	m, n = size(p)

	max_hi = -Inf
	max_index = 0

	for i in 1:m
		dot = sum((x[j] - p[i, j])^2 for j in 1:n)
		if dot > max_hi
			max_hi = dot
			max_index = i
		end
	end

	h = sum((x[j] - p[max_index, j])^2 for j in 1:n)

	if h >= 0.0
		grad = x .- view(p, max_index, :)
		norm_grad = sum(abs2, grad)
		if norm_grad <= 0.0
			println("update a3pm has too small norm $norm_grad")
		end
		step_size = h / norm_grad
		step_size = clamp(step_size, 1e-6, 1.0)
		x .-= step_size .* grad
	end
	x .= (1.0 / iter) * x0 + (1.0 - (1.0 / iter)) * x
	return x
end

"""
update_alt_proj!(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray, k::AbstractVector,
	env::Gurobi.Env)

Updates the current solution using alternating projections

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real:              precision parameter
"""
function update_alt_proj_halpern!(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray, env::Gurobi.Env, iter::Int, x0::AbstractVector)
	k, _, n = size(Q)

	for i in 1:k
		x .= polyhedron_projection(x, y[i, :], Q[i, :, :], env)
	end
	x .= (1.0 / iter) * x0 + (1.0 - (1.0 / iter)) * x
	return x
end


function solve_dijkstra(x0::AbstractVector, epsilon::Float64, p::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env,
	target::AbstractVector)
	k, _, n = size(Q)
	yprev = zeros(k, n)
	xprev = copy(x0)
	x = zeros(k, n)
	y = zeros(k, n)
	violation = []
	t_start = time()
	while (norm(xprev - target) > epsilon)
		for i in 1:k
			if (i == 1)
				x[i, :] = polyhedron_projection(xprev - yprev[i, :], p[i, :], Q[i, :, :], env)
			else
				x[i, :] = polyhedron_projection(x[i-1, :] - yprev[i, :], p[i, :], Q[i, :, :], env)
			end
		end
		for i in 1:k
			if (i == 1)
				y[i, :] = x[i, :] - (xprev - yprev[i, :])
			else
				y[i, :] = x[i, :] - (x[i-1, :] - yprev[i, :])
			end
		end
		yprev .= y
		xprev .= x[k, :]
		current_violation = max_violation(xprev, p, Q)
		push!(violation, current_violation)
		t_current = time() - t_start
		if t_current > 600.0
			break
		end
	end

	return xprev, violation, length(violation), time() - t_start
end

"""
update_cimmino!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Updates the current solution using cimmino's projection
- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function update_cimmino_halpern!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, iter::Int, x0::AbstractVector)

	k, _, n = size(Q)
	sum_y = zeros(Float64, n)

	for i in 1:k
		if all(Q[i, :, :] * x .- y[i, :] .<= 0.0)
			sum_y .+= x
		else
			sum_y .+= polyhedron_projection(x, y[i, :], Q[i, :, :], env)
		end
	end

	x .= (1.0 / iter) * x0 + (1.0 - (1.0 / iter)) * (1.0 / k) .* sum_y

	return x
end

"""
update_cimmino_mt!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Updates the current solution using cimmino's projection in parallel
- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function update_cimmino_halpern_mt!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, iter::Int, x0::AbstractVector)

	k, _, n = size(Q)
	p = zeros(Float64, k, n)

	Threads.@threads for i in 1:k
		if all(Q[i, :, :] * x .- y[i, :] .<= 0.0)
			p[i, :] .= x
		else
			p[i, :] .= polyhedron_projection(x, y[i, :], Q[i, :, :], env)
		end
	end
	# @show size(x0)
	# @show size(sum(p, dims = 1)')
	x .= (1.0 / iter) * x0 + (1.0 - (1.0 / iter)) * (1.0 / k) .* sum(p, dims = 1)'
end

"""
composite_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)

Computes the composite projection P_A(P_B(x))
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter

"""
function composite_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env)
	return polyhedron_projection(polyhedron_projection(x, y[B, :], Q[B, :, :], env), y[A, :], Q[A, :, :], env)
end

"""
average_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, ϵ::Real)

Computes the average projection of A and B: Z = P_A(x) + P_B(x)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function average_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env)
	return 0.5 * (polyhedron_projection(x, y[A, :], Q[A, :, :], env) +
				  polyhedron_projection(x, y[B, :], Q[B, :, :], env))
end

"""
find_circuncenter(x::AbstractMatrix)

Given m points x ∈ ℜ^{n}, finds the equidistant point to all points x.
- x::AbstractMatrix:    Set of points x ∈ ℜ^{n}
"""
function find_circuncenter(x::AbstractMatrix)
	m = size(x, 1) - 1
	M = similar(x, m, m)
	b = similar(x, m)
	x_0 = x[1, :]
	for i in 1:m
		for j in 1:m
			M[i, j] = dot(x[j+1, :] .- x_0, x[i+1, :] .- x_0)
		end
		b[i] = 0.5 * norm(x[i+1, :] .- x_0)^2
	end
	result = similar(x_0)
	try
		α = M \ b
		result = copy(x_0)
		for j in 1:m
			result .+= α[j] .* (x[j+1, :] .- x_0)
		end
	catch e
		if isa(e, SingularException) || isa(e, PosDefException)
			result .= sum(x, dims = 1)[:] ./ size(x, 1)
		else
			rethrow(e)
		end
	end
	return result
end


"""
Z(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env)

	Computes the projection operator ̄z(x) = average_projection(A,B) + composite_projection(A,B)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function Z(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env)
	return average_projection(composite_projection(x, A, B, y, Q, env), A, B, y, Q, env)
end

"""
reflection(x::AbstractVector, projection::AbstractVector)

Computes the reflection of x given a projected point

- x::AbstractVector:                current point
- projection::AbstractVector:       projected point
"""
function reflection(x::AbstractVector, projection::AbstractVector)
	if size(x) != size(projection)
		return error("Dimension mismatch at reflection: size(x) = $(size(x)), size(projection) = $(size(projection))")
	end
	return 2 * projection .- x
end

function update_sc_crm_halpern!(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, iter::Int, x0::AbstractVector)
	z_bar = Z(x, A, B, y, Q, env)

	R_A = reflection(z_bar, polyhedron_projection(z_bar, y[A, :], Q[A, :, :], env))
	R_B = reflection(z_bar, polyhedron_projection(z_bar, y[B, :], Q[B, :, :], env))

	k, _, n = size(Q)
	P = zeros(3, n)
	P[1, :] .= z_bar
	P[2, :] .= R_A
	P[3, :] .= R_B
	x .= find_circuncenter(P)
	x .= (1 / iter) * x0 + (1 - (1 / iter)) * x
	return x
end

"""
average_product_space_projection(z::AbstractVector, m::UInt, n::UInt)

Computes the average projection in the product space.
- z::AbstractVector     Current solution in product-space, z ∈ ℜ^{mn}
- m::Uint               size parameter for z 
- n::Uint               size parameter for z
"""
function average_product_space_projection(z::AbstractVector, m::Int, n::Int)
	if size(z, 1) != m * n
		return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
	end
	Y = reshape(z, n, m)
	return (1.0 / m) * repeat(vec(sum(Y, dims = 2)), m)
end


"""
elipsoid_product_space_projection(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Computes the projection of z ∈ ℜ^{mn} into the product space of the elipsoids (y_i,Q_i,k_i), i in 1:m
- z::AbstractVector:        Current solution in product-space, z ∈ ℜ^{mn} 
- y::AbstractMatrix:        Elipsoid centers 
- Q::AbstractArray:         Elipsoid matrices
- k::AbstractVector:        Elipsoid radii
- env::Gurobi.Env:          Gurobi environment to solve projections
"""
function polyhedron_product_space_projection(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env)
	k, _, n = size(Q)

	if size(z, 1) != k * n
		return error("z has wrong size $(size(z,1)) != m*n = $(k*n)")
	end

	Y = reshape(z, n, k)
	Z = similar(Y)
	for i in 1:k
		Z[:, i] .= polyhedron_projection(Y[:, i], y[i, :], Q[i, :, :], env)
	end
	return vec(Z)
end


function update_crm!(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env)
	k, _, n = size(Q)
	if size(z, 1) != k * n
		return error("z has wrong size $(size(z,1)) != m*n = $(k*n)")
	end
	P = zeros(3, k * n)
	P[1, :] .= z
	P[2, :] .= reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
	R_W = reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
	P[3, :] .= reflection(R_W, average_product_space_projection(R_W, k, n))
	z .= find_circuncenter(P)
	return z
end


function update_crm_halpern!(z::AbstractVector, z0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, iter::Int)
	k, _, n = size(Q)
	if size(z, 1) != k * n
		return error("z has wrong size $(size(z,1)) != m*n = $(k*n)")
	end
	P = zeros(3, k * n)
	P[1, :] .= z
	P[2, :] .= reflection(z, polyhedron_product_space_projection(z, y, Q, env))
	R_W = reflection(z, polyhedron_product_space_projection(z, y, Q, env))
	P[3, :] .= reflection(R_W, average_product_space_projection(R_W, k, n))
	z .= find_circuncenter(P)
	z .= (1 / iter) * z0 .+ (1 - (1 / iter)) * z
	return z
end

"""
solve_3pm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)

Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i)

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- env::Gurobi.Env:      Gurobi environment to solve projection and/or update problems
- ϵ::Real               precision parameter
"""

function solve_3pm_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray, max_iter::UInt, env::Gurobi.Env, ϵ::Real, target::AbstractVector)
	x = copy(x0)
	k, m, n = size(Q)
	violation = []
	iter = 1
	t_start = time()
	while (norm(x - target) > ϵ)
		p = mt ? projection_3pm_mt(x, y, Q, env) : projection_3pm(x, y, Q, env)
		update_3pm_halpern!(x, p, env, ϵ, iter, x0)
		# push!(points_x, x[1])
		# push!(points_y, x[2])
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		if current_violation <= 0.0
			break
		end
		t_current = time() - t_start
		if t_current > 600.0
			break
		end
		iter += 1
	end
	return x, violation, iter, time() - t_start
end



function solve_3pm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env, ϵ::Real)
	x = copy(x0)
	k, m, n = size(Q)
	violation = []
	iter = 1
	while iter < max_iter
		p = mt ? projection_3pm_mt(x, y, Q, env) : projection_3pm(x, y, Q, env)
		update_3pm!(x, p, env, ϵ, iter, x0)
		# push!(points_x, x[1])
		# push!(points_y, x[2])
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		if current_violation <= 0.0
			break
		end
		iter += 1
	end
	return x, violation
end


"""
solve_a3pm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt)

Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using the a3pm projections

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- ϵ::Real               precision parameter
"""
function solve_a3pm_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	epsilon::Real, target::AbstractVector, mt::Bool = false)
	x = copy(x0)
	m, n = size(y)
	iter = 1
	violation = []
	t_start = time()
	while (norm(x - target) > epsilon)
		p = mt ? projection_a3pm_mt(x, y, Q) : projection_a3pm(x, y, Q)
		update_a3pm_halpern!(x, p, iter, x0)
		# push!(points_x, x[1])
		# push!(points_y, x[2])
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		# println("Iter ", iter, ", x = ", x, " p = ", p)
		# if current_violation <= 0.0
		#     break
		# end
		iter += 1
		t_current = time() - t_start
		if t_current > 600.0
			break
		end
	end
	return x, violation, iter, time() - t_start

end

"""
solve_alt_proj(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using alternating projections

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter
"""
function solve_alt_proj_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, epsilon::Real, target::AbstractVector)
	x = copy(x0)
	iter = 1
	violation = []

	t_start = time()
	while (norm(x - target) > epsilon)
		update_alt_proj_halpern!(x, y, Q, env, iter, x0)
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		iter += 1
		t_current = time() - t_start
		if t_current > 600.0
			break
		end
	end
	return x, violation, iter, time() - t_start
end


"""
solve_cimmino(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using cimmino projections

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter
"""
function solve_cimmino_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray, env::Gurobi.Env,
	epsilon::Real, target::AbstractVector, mt::Bool = false)
	x = copy(x0)
	iter = 1
	violation = []

	t_start = time()
	while (norm(x - target) > epsilon)
		if mt
			update_cimmino_halpern_mt!(x, y, Q, env, iter, x0)
		else
			update_cimmino_halpern!(x, y, Q, env, iter, x0)
		end
		# push!(points_x, x[1])
		# push!(points_y, x[2])
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		iter += 1
		# if current_violation <= 0.0
		#     break
		# end
		t_current = time() - t_start
		if t_current > 600.0
			println("Time limit exceeded")
			break
		end
	end
	return x, violation, iter, time() - t_start
end


"""
solve_sc_crm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using successive centralized CRMs

"""
function solve_sc_crm_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, epsilon::Real, target::AbstractVector)
	x = copy(x0)
	k, _, _ = size(Q)
	iter = 1
	violation = []

	t_start = time()
	while (norm(x - target) > epsilon)
		A::UInt = (iter % k) + 1
		B::UInt = A == k ? 1 : A + 1
		update_sc_crm_halpern!(x, A, B, y, Q, env, iter, x0)
		# push!(points_x, x[1])
		# push!(points_y, x[2])
		iter += 1
		current_violation = max_violation(x, y, Q)
		push!(violation, current_violation)
		# if current_violation <= 0.0
		#     break
		# end
		t_current = time() - t_start
		if t_current > 600.0
			# println("\tTime limit exceeded")
			break
		end
	end
	return x, violation, iter, time() - t_start
end


function solve_crm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	max_iter::UInt, env::Gurobi.Env, ϵ::Real)
	x = copy(x0)
	k, m, n = size(y)
	iter = 1
	z = repeat(x, k)
	violation = []
	while iter < max_iter
		update_crm!(z, y, Q, env)
		iter += 1
		Y = reshape(z, n, k)
		current_violation = -Inf
		current_violation_index = -1
		for i in 1:k
			violation_i = max_violation(Y[:, i], y, Q)
			if violation_i > current_violation
				current_violation = violation_i
				current_violation_index = i
			end
		end

		push!(violation, current_violation)
		if current_violation <= 0.0
			break
		end
	end

	return z, violation, iter
end

function solve_crm_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	env::Gurobi.Env, epsilon::Real, target::AbstractVector)
	x = copy(x0)
	k, _, n = size(Q)
	iter = 1
	z = repeat(x, k)
	z0 = copy(z)
	violation = []
	t_start = time()
	z_target = repeat(target, k)
	while (norm(z - z_target) > epsilon)
		update_crm_halpern!(z, z0, y, Q, env, iter)
		iter += 1
		Y = reshape(z, n, k)
		current_violation = -Inf
		current_violation_index = -1
		for i in 1:k
			violation_i = max_violation(Y[:, i], y, Q)
			if violation_i > current_violation
				current_violation = violation_i
				current_violation_index = i
			end
		end
		push!(violation, current_violation)
		t_current = time() - t_start
		if t_current > 600.0
			break
		end
		# if current_violation <= 0.0
		#     break
		# end
	end

	return z, violation, iter, time() - t_start
end

function test_cutting_plane2(n::UInt)
	env = Gurobi.Env()
	Q = zeros(240, 1, n)
	y = zeros(240, 1)
	Random.seed!(0)
	x_star = randn(n)
	for i in 1:200
		A = randn(1, n)
		b = A * x_star
		Q[i, :, :] .= A
		y[i, :] .= b
	end

	for i in 201:220
		a = normalize(Q[i, :, :] + 0.001 * randn(1, n))
		Q[i, :, :] .= a
		y[i, :] .= a * x_star
	end

	for i in 221:240
		ind = rand(1:200)
		aux = Q[ind, :, :] + 1e-2 * randn(1, n)
		Q[i, :, :] .= normalize(aux)
		y[i, :] .= aux * x_star
	end
	x0 = 20 * ones(1, n)
	ϵ = 0.01
	println("\tviolation x0 = ", max_violation(x0, y, Q))
	# target = projection_intersection_polyhedron(x0, y, Q, 0.1, env)
	target = projection_intersection_polyhedron(x_star, y, Q, 0.1, env)
	println("\tTarget found!")

	# x_a3pm, violation_a3pm, iter_a3pm, time_a3pm = solve_a3pm_halpern(x0, y, Q, k, ϵ, target)
	x_a3pm, violation_a3pm, iter_a3pm, time_a3pm = solve_a3pm_halpern(x0, y, Q, ϵ, target)
	@show violation_a3pm[end], time_a3pm, iter_a3pm


	# time_alt_proj = @elapsed x_alt_proj, violation_alt, iter_alt = solve_alt_proj_halpern(x0, y, Q, k, env, 0.1, target)
	x_alt_proj, violation_alt, iter_alt, time_alt_proj = solve_alt_proj_halpern(x0, y, Q, env, ϵ, target)
	@show violation_alt[end], time_alt_proj, iter_alt

	# time_cimmino = @elapsed x_cimmino, violation_cimmino, iter_cimmino = solve_cimmino_halpern(x0, y, Q, k, env, ϵ, target)
	x_cimmino, violation_cimmino, iter_cimmino, time_cimmino = solve_cimmino_halpern(x0, y, Q, env, ϵ, target)
	@show violation_cimmino[end], time_cimmino, iter_cimmino

	# x_sccrm, violation_sccrm, iter_sccrm, time_sccrm = solve_sc_crm_halpern(x0, y, Q, k, env, 0.1, target)
	x_sccrm, violation_sccrm, iter_sccrm, time_sccrm = solve_sc_crm_halpern(x0, y, Q, env, ϵ, target)
	@show violation_sccrm[end], time_sccrm, iter_sccrm

	x_crm, violation_crm, iter_crm, time_crm = solve_crm_halpern(x0, y, Q, env, ϵ, target)
	@show violation_crm[end], time_crm, iter_crm

	# x_dijkstra, violation_dijkstra, iter_dijkstra, time_dijkstra = solve_dijkstra(x0, 0.1, y, Q, k, env, target)
	x_dijkstra, violation_dijkstra, iter_dijkstra, time_dijkstra = solve_dijkstra(x0, ϵ, y, Q, env, target)
	@show violation_dijkstra[end], time_dijkstra, iter_dijkstra

	println("\tA3PM: $time_a3pm s, iter = $iter_a3pm, violation = $(violation_a3pm[end])")
	println("\tAlt Proj: $time_alt_proj s, iter = $iter_alt, violation = $(violation_alt[end])")
	println("\tCimmino: $time_cimmino s, iter = $iter_cimmino, violation = $(violation_cimmino[end])")
	println("\tSC CRM: $time_sccrm s, iter = $iter_sccrm, violation = $(violation_sccrm[end])")
	println("\tDijkstra: $time_dijkstra s, iter = $iter_dijkstra, violation = $(violation_dijkstra[end])")
	println("\tCRM: $time_crm s, iter = $iter_crm, violation = $(violation_crm[end])")
end

using BenchmarkTools

function test_cutting_plane(k::UInt, n::UInt, m::UInt, max_iter::UInt, alpha::Real)
	env = Gurobi.Env()
	y = zeros(k, m)
	Q = zeros(k, m, n)
	Random.seed!(0)
	x_star = randn(n)

	for i in 1:k
		A = randn(m, n)
		b = A * x_star + alpha * rand(m)
		Q[i, :, :] .= A
		y[i, :] .= b
	end

	x0 = 20 * ones(n)
	ϵ = 0.1
	println("\tviolation x0 = ", max_violation(x0, y, Q))
	target = projection_intersection_polyhedron(x0, y, Q, 0.1, env)
	println("\tTarget found!")



  
    @info "Running A3PM..."
    # x_a3pm, violation_a3pm, iter_a3pm, time_a3pm = solve_a3pm_halpern(x0, y, Q, k, ϵ, target)
    x_a3pm, violation_a3pm, iter_a3pm, _ = solve_a3pm_halpern(x0, y, Q, ϵ, target)
    time_a3pm = @belapsed solve_a3pm_halpern($x0, $y, $Q, $ϵ, $target)
    @show violation_a3pm[end], time_a3pm, iter_a3pm

    @info "Running Alternating Projections..."
    # time_alt_proj = @elapsed x_alt_proj, violation_alt, iter_alt = solve_alt_proj_halpern(x0, y, Q, k, env, 0.1, target)
    x_alt_proj, violation_alt, iter_alt, _  = solve_alt_proj_halpern(x0, y, Q, env, ϵ, target)
    time_alt_proj = @belapsed solve_alt_proj_halpern($x0, $y, $Q, $env, $ϵ, $target)
    @show violation_alt[end], time_alt_proj, iter_alt

    @info "Running Cimmino..."
    # time_cimmino = @elapsed x_cimmino, violation_cimmino, iter_cimmino = solve_cimmino_halpern(x0, y, Q, k, env, ϵ, target)
    x_cimmino, violation_cimmino, iter_cimmino, _ = solve_cimmino_halpern(x0, y, Q, env, ϵ, target)
    time_cimmino = @belapsed solve_cimmino_halpern($x0, $y, $Q, $env, $ϵ, $target)
    @show violation_cimmino[end], time_cimmino, iter_cimmino

    @info "Running SC CRM..."
    # x_sccrm, violation_sccrm, iter_sccrm, time_sccrm = solve_sc_crm_halpern(x0, y, Q, k, env, 0.1, target)
    x_sccrm, violation_sccrm, iter_sccrm, _ = solve_sc_crm_halpern(x0, y, Q, env, ϵ, target)
    time_sccrm = @belapsed solve_sc_crm_halpern($x0, $y, $Q, $env, $ϵ, $target)
    @show violation_sccrm[end], time_sccrm, iter_sccrm

    @info "Running CRM..."
    x_crm, violation_crm, iter_crm, _ = solve_crm_halpern(x0, y, Q, env, ϵ, target)
    time_crm = @belapsed solve_crm_halpern($x0, $y, $Q, $env, $ϵ, $target)
    @show violation_crm[end], time_crm, iter_crm

    # x_dijkstra, violation_dijkstra, iter_dijkstra, time_dijkstra = solve_dijkstra(x0, 0.1, y, Q, k, env, target)
    x_dijkstra, violation_dijkstra, iter_dijkstra, _ = solve_dijkstra(x0, ϵ, y, Q, env, target)
    @info "Running Dijkstra..."
    time_dijkstra = @belapsed solve_dijkstra($x0, $ϵ, $y, $Q, $env, $target)
    @show violation_dijkstra[end], time_dijkstra, iter_dijkstra

	println("\tA3PM: $time_a3pm s, iter = $iter_a3pm, violation = $(violation_a3pm[end])")
	println("\t3PM: $time_3pm s, iter = $iter_3pm")
	println("\tAlt Proj: $time_alt_proj s, iter = $iter_alt, violation = $(violation_alt[end])")
	println("\tCimmino: $time_cimmino s, iter = $iter_cimmino, violation = $(violation_cimmino[end])")
	println("\tSC CRM: $time_sccrm s, iter = $iter_sccrm, violation = $(violation_sccrm[end])")
	println("\tDijkstra: $time_dijkstra s, iter = $iter_dijkstra, violation = $(violation_dijkstra[end])")
	println("\tCRM: $time_crm s, iter = $iter_crm, violation = $(violation_crm[end])")


	# savefig(p2, "violations.pdf")
end

function test_find_circuncenter(m, n)
	Random.seed!(0)

	# Gerar 3 pontos aleatórios no plano (triângulo)
	x = rand(m, n)

	# Calcular circuncentro
	c = find_circuncenter(x)
	@show c
	@show norm(c .- x[1, :])
	@show norm(c .- x[2, :])
	@show norm(c .- x[3, :])
	if n > 2
		return
	end

	# # Plot
	# p2 = plot()
	# scatter!(p2, x[:, 1], x[:, 2], label="Points", legend=:topright, ms=8, markerstrokewidth=0)
	# scatter!(p2, [c[1]], [c[2]], label="Circuncenter", ms=10, marker=:star5, color=:red)
	# # Também desenhar linhas ligando os pontos
	# plot!(p2, [x[:, 1]; x[1, 1]], [x[:, 2]; x[1, 2]], label="", lw=1, ls=:dash, color=:gray)

	# title!(p2, "Circuncenter")
	# xlabel!(p2, "x₁")
	# ylabel!(p2, "x₂")
	# savefig(p2, "circuncenter.pdf")
end

function main()
	k::UInt = 10
	n::UInt = 100
	m::UInt = 100
	max_iter::UInt = 100
	alphas = [1.0]
	ns = [10, 10, 10, 10, 20, 100]
	ms = [10, 10, 10, 10, 20, 100]
	ks = [20, 10, 5, 3, 20, 20]


	# for i in eachindex(ns)
	#     n = ns[i]
	#     println("Testing n = $n")
	#     test_cutting_plane2(n)
	# end

	for alpha in alphas
		for i in eachindex(ns)
			k = ks[i]
			n = ns[i]
			m = ms[i]
			println("Testing alpha = $alpha n = $n, m = $m, k = $k")
			test_cutting_plane(k, n, m, max_iter, alpha)
		end
	end
	# sleep(1)
	# test_find_circuncenter(m, n)
end

main()
