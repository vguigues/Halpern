using Gurobi
using JuMP
using LinearAlgebra
using Random
using Base.Threads
using LaTeXStrings

include("plot.jl")
include("util.jl")

println("Num threads: ", nthreads())

using Printf

const EPS = 1e-5
const mt = false 

function projection_cone_sdp(X::AbstractMatrix)

X.=(X+X')/2    
m,n=size(X)
result = eigen(X)
U=result.vectors
s=real.(result.values)
# ipart=imag.(result.values)
# println("Value of eigenvalue r part: $s\n")
# println("Value of eigenvalue i part: $ipart\n")main




aux=zeros(n)
for i in 1:n
    aux[i]=max(0,s[i]) 
end

return U*Diagonal(aux)*U'
    
end


function projection_affine(X::AbstractMatrix,A::AbstractArray,m::Int,b::AbstractVector)


G=zeros(m,m)
r=zeros(m)
lambda=zeros(m)

for i in 1:m 
    for j in 1:m
        G[i,j]=dot(A[i,:,:],A[j,:,:])
    end
    r[i]=dot(A[i,:,:],X)-b[i]
end
lambda=G\r

Mat=zeros(m,m)
Mat.=X
for i in 1:m 
    Mat.=Mat-lambda[i]*A[i,:,:]
end

return Mat

end

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
function projection_3pm(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector, env::Gurobi.Env)
    m, n = size(Q, 1), size(Q, 2)
    p = zeros(m, n)

    for i in 1:m
        p[i, :] = elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
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
function projection_3pm_mt(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector, env::Gurobi.Env)
    m, n = size(Q, 1), size(Q, 2)
    p = zeros(m, n)

    Threads.@threads for i in 1:m
        p[i, :] .= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
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
    Q::AbstractArray, k::AbstractVector)

    m, n = size(Q, 1), size(Q, 2)
    p = zeros(eltype(x), m, n)

    for i in 1:m
        g_i = g(x, y[i, :], Q[i, :, :], k[i])
        if g_i <= 0.0
            p[i, :] .= x
        else
            grad = g_prime(x, y[i, :], Q[i, :, :])
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
    Q::AbstractArray, k::AbstractVector)

    m, n = size(Q, 1), size(Q, 2)
    p = zeros(eltype(x), m, n)

    Threads.@threads for i in 1:m
        g_i = g(x, y[i, :], Q[i, :, :], k[i])
        if g_i <= 0.0
            p[i, :] .= x
        else
            grad = g_prime(x, y[i, :], Q[i, :, :])
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
function update_3pm!(x::AbstractVector, p::AbstractMatrix, env::Gurobi.Env, ϵ::Real,iter::Int,xo::AbstractVector)
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

"""
update_a3pm!(x::AbstractVector, p::AbstractMatrix)

Updates x using the a3pm projections p.
-  x::AbstractVector:   size(x) = n, current solution
-  p::AbstractMatrix:   size(p) = (m,n), 3pm projections
-  ϵ::Real:             precision parameter
"""
function update_a3pm_halpern!(x::AbstractVector, p::AbstractMatrix,iter::Int,x0::AbstractVector)
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
    x.=(1.0/iter)*x0+(1.0-(1.0/iter))*x
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

function update_alt_proj_halpern!(X::AbstractMatrix,X0::AbstractMatrix,A::AbstractArray,n::Int,b::AbstractVector,iter::Int)
    
X.=projection_cone_sdp(X)
X.=projection_affine(X,A,n,b)
X.=(1.0/(1.5*iter))*X0+(1.0-(1.0/(1.5*iter)))*X

return X

end

function dijkstra(x0::AbstractMatrix,n::Int,A::AbstractArray,m::Int,b::AbstractVector,max_iter::Int)

start_time = time()
iter=1
yprev=zeros(2,n,n)
xprev=copy(x0)
x=zeros(m,n,n)
y=zeros(m,n,n)
while (iter<=max_iter)
    for i in 1:m
        if (i==1)
           x[i,:,:]=projection_cone_sdp(xprev-yprev[i, :,:])
        else
           x[i,:,:]=projection_affine(x[i-1, :,:]-yprev[i, :,:],A,n,b)
        end 
    end
    for i in 1:m
        if (i==1)
           y[i,:,:]=x[i, :,:]-(xprev-yprev[i, :,:])
        else
           y[i,:,:]=x[i, :,:]-(x[i-1, :,:]-yprev[i, :,:])
        end 
    end
    yprev.=y 
    xprev.=x[m,:,:]
    iter+=1
    println("Iteration: $iter")
end
elapsed = time() - start_time
return xprev,elapsed
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
    k::AbstractVector, env::Gurobi.Env,iter::Int,x0::AbstractVector)

    m, n = size(Q, 1), size(Q, 2)
    sum_y = zeros(Float64, n)

    for i in 1:m
        if g(x, y[i, :], Q[i, :, :], k[i]) <= 0.0
            sum_y .+= x
        else
            sum_y .+= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
        end
    end

    x .= (1.0/iter)*x0+(1.0-(1.0/iter))*(1.0 / m) .* sum_y

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
    k::AbstractVector, env::Gurobi.Env,iter::Int,x0::AbstractVector)

    m, n = size(Q, 1), size(Q, 2)
    p = zeros(Float64, m, n)

    Threads.@threads for i in 1:m
        if g(x, y[i, :], Q[i, :, :], k[i]) <= 0.0
            p[i, :] .= x
        else
            p[i, :] .= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
        end
    end

    x .= (1.0/iter)*x0+(1.0-(1.0/iter))*(1.0 / m).* sum(p, dims=1)
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

function composite_projection(X::AbstractMatrix, A::AbstractArray,m::Int,b::AbstractVector)
    return projection_cone_sdp(projection_affine(X,A,m,b))
end

"""
average_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env, ϵ::Real)

Computes the average projection of A and B: Z = P_A(x) + P_B(x)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function average_projection(X::AbstractVector, A::AbstractArray,m::Int,b::AbstractVector)
    return 0.5 * (projection_cone_sdp(A)+
                  projection_affine(X,A,m,b))
end

"""
find_circuncenter(x::AbstractMatrix)

Given m points x ∈ ℜ^{n}, finds the equidistant point to all points x.
- x::AbstractMatrix:    Set of points x ∈ ℜ^{n}
"""


function find_circuncenter2(x::AbstractArray)
    m = size(x, 1) - 1
    M = similar(x, m, m)
    b = similar(x, m)
    x_0 = x[1, :,:]
    for i in 1:m
        for j in 1:m
            M[i, j] = dot(x[j+1, :,:] .- x_0, x[i+1, :,:] .- x_0)
        end
        b[i] = 0.5*dot(x[i+1, :,:] .- x_0,x[i+1, :,:] .- x_0)
    end
    result = similar(x_0)
    α = M \ b
    result = copy(x_0)
    for j in 1:m
        result .+= α[j] .* (x[j+1, :,:] .- x_0)
    end 
    return result
end

function find_circuncenter(x::AbstractArray)
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
            result .= sum(x, dims=1)[:] ./ size(x, 1)
        else
            rethrow(e)
        end
    end
    return result
end


"""
Z(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env)

    Computes the projection operator ̄z(x) = average_projection(A,B) + composite_projection(A,B)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function Z(X::AbstractMatrix, A::AbstractArray,m::Int,b::AbstractVector)
    return average_projection(composite_projection(X,A,m,b),X,A,m,b)
end

"""
reflection(x::AbstractVector, projection::AbstractVector)

Computes the reflection of x given a projected point

- x::AbstractVector:                current point
- projection::AbstractVector:       projected point
"""
function reflection(x::AbstractMatrix, projection::AbstractMatrix)
    return 2 * projection .- x
end

function update_sc_crm_halpern!(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,iter::Int,x0::AbstractVector)
    z_bar = Z(x, A, B, y, Q, k, env)

    R_A = reflection(z_bar,projection_cone_sdp(A))
    R_B = reflection(z_bar,projection_affine(X,A,m,b))

    _, n = size(y)
    P = zeros(3, n)
    P[1, :] .= z_bar
    P[2, :] .= R_A
    P[3, :] .= R_B
    x .= find_circuncenter(P)
    x.=(1/iter)*x0+(1-(1/iter))*x
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
    return (1.0 / m) * repeat(vec(sum(Y, dims=2)), m)
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
function elipsoid_product_space_projection(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env)
    m, n = size(y)

    if size(z, 1) != m * n
        return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
    end

    Y = reshape(z, n, m)
    Z = similar(Y)
    for i in 1:m
        Z[:, i] .= elipsoid_projection(Y[:, i], y[i, :], Q[i, :, :], k[i], env)
    end
    return vec(Z)
end


function update_crm2!(X0::AbstractMatrix,z::AbstractMatrix,A::AbstractArray,m::Int,b::AbstractVector,iter::Int)
    P = zeros(3,m,m)
    P[1, :,:] .= z
    P[2, :,:] .= reflection(z,projection_cone_sdp(z))
    P[3, :,:] .= reflection(P[2, :,:],projection_affine(P[2, :,:],A,m,b))
    z .= find_circuncenter2(P)
    z.=(1.0/(1.5*iter))*X0+(1.0-(1.0/(1.5*iter)))*z
    return z
end


function update_crm!(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env)
    m, n = size(y)
    if size(z, 1) != m * n
        return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
    end
    P = zeros(3, m * n)
    P[1, :] .= z
    P[2, :] .= reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
    R_W = reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
    P[3, :] .= reflection(R_W, average_product_space_projection(R_W, m, n))
    z .= find_circuncenter(P)
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
function solve_3pm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, env::Gurobi.Env, ϵ::Real)
    x = copy(x0)
    m, n = size(y)
    violation = []
    iter = 1
    p1 = plot(aspect_ratio=1, label="3pm")
    if n == 2
        plot_all_elipsoids(p1, y, Q, k)
        plot_1ball(p1)
        scatter!(p1, [x0[1]], [x0[2]],
            markershape=:circle,
            label="x0",
            color=:orange,
            markersize=5)
    end
    points_x = []
    points_y = []
    while iter < max_iter
        p = mt ? projection_3pm_mt(x, y, Q, k, env) : projection_3pm(x, y, Q, k, env)
        update_3pm!(x, p, env, ϵ,iter,x0)
        push!(points_x, x[1])
        push!(points_y, x[2])
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)
        if current_violation <= 0.0
            break
        end
        iter += 1
    end
    println("3pm iter = $iter")
    if n == 2
        scatter!(p1, points_x, points_y, markershape=:circle,
            label="x_3pm", color=:blue, markersize=3)
        scatter!(p1, [x[1]], [x[2]],
            markershape=:circle,
            label="x_3pm_final",
            color=:red,
            markersize=3)
        legend_str = iter == 1 ? "iteration" : "iterations"
        title!(p1, "3PM ($iter $legend_str)")
        savefig(p1, "projections_3pm.pdf")
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
function solve_a3pm_halpern(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector,
    max_iter::UInt, ϵ::Real,target::AbstractVector)
    x = copy(x0)
    m, n = size(y)
    iter = 1

    violation = []
    p1 = plot(aspect_ratio=1, label="3pm")
    if n == 2
        plot_all_elipsoids(p1, y, Q, k)
        plot_1ball(p1)
        scatter!(p1, [x0[1]], [x0[2]],
            markershape=:circle,
            label="x0",
            color=:orange,
            markersize=5)
    end
    points_x = []
    points_y = []
    while (norm(x-target)>ϵ)
        p = mt ? projection_a3pm_mt(x, y, Q, k) : projection_a3pm(x, y, Q, k)
        update_a3pm_halpern!(x, p,iter,x0)
        push!(points_x, x[1])
        push!(points_y, x[2])
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)
        # println("Iter ", iter, ", x = ", x, " p = ", p)
        if current_violation <= 0.0
            break
        end
        iter += 1
    end
    if n == 2
        scatter!(p1, points_x, points_y, markershape=:circle,
            label="x_a3pm", color=:blue, markersize=3)
        scatter!(p1, [x[1]], [x[2]],
            markershape=:circle,
            label="x_a3pm_final",
            color=:red,
            markersize=3)
        legend_str = iter == 1 ? "iteration" : "iterations"
        title!(p1, "A3PM ($iter $legend_str)")
        savefig(p1, "projections_a3pm.pdf")
    end
    println("a3pm iter = $iter")
    return x, violation
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


function solve_alt_proj_halpern(C::AbstractMatrix,A::AbstractArray,b::AbstractVector,n::Int,max_iter::Int,epsilon::Float64,target::AbstractMatrix)
    
    start_time = time()
    X = copy(C)
    Xs = Vector{Matrix{Float64}}()
    iter = 1
    while (sqrt(dot(target-X,target-X))>epsilon)
        println("$(sqrt(dot(target-X,target-X)))")
        # while (iter<=max_iter)
        # println("Iteration: $iter")
        X=update_alt_proj_halpern!(X,C,A,n,b,iter)
        push!(Xs,X)
        iter += 1
        println("Iteration: $iter")
    end
    elapsed=time()-start_time
    return Xs,elapsed
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
function solve_cimmino(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, env::Gurobi.Env, ϵ::Real)
    x = copy(x0)
    iter = 0
    m, n = size(y)
    violation = []
    p1 = plot(aspect_ratio=1)
    if n == 2
        plot_all_elipsoids(p1, y, Q, k)
        plot_1ball(p1)
        scatter!(p1, [x0[1]], [x0[2]],
            markershape=:circle,
            label="x0",
            color=:orange,
            markersize=5)
    end
    points_x = []
    points_y = []
    while iter < max_iter
        if mt
            update_cimmino_halpern_mt!(x, y, Q, k, env,iter,x0)
        else
            update_cimmino_halpern!(x, y, Q, k, env,iter,x0)
        end
        push!(points_x, x[1])
        push!(points_y, x[2])
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)
        iter += 1
        if current_violation <= 0.0
            break
        end
    end
    if n == 2
        scatter!(p1, points_x, points_y,
            markershape=:circle,
            label="x_cimmino",
            color=:blue,
            markersize=3)
        scatter!(p1, [x[1]], [x[2]],
            markershape=:circle,
            label="x_cimmino_final",
            color=:red,
            markersize=3)
        legend_str = iter == 1 ? "iteration" : "iterations"
        title!(p1, "Cimmino method ($iter $legend_str)")
        savefig(p1, "projections_cimmino.pdf")
    end
    println("cimmino iter = $iter")
    return x, violation
end

"""
solve_sc_crm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using successive centralized CRMs

"""


function solve_sc_crm_halpern(x0::AbstractVector,A::AbstractArray,b::AbstractVector,max_iter::UInt, ϵ::Real)
    x = copy(x0)
    iter = 0
    violation = []
    points_x = []
    points_y = []
    while iter < max_iter
        update_sc_crm_halpern!(x,A,x0)
        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)
        if current_violation <= 0.0
            break
        end
    end

    return x, violation
end



function solve_crm2(C::AbstractMatrix,A::AbstractArray,b::AbstractVector,n::Int,epsilon::Float64,target::AbstractMatrix)
    x = copy(C)
    iter = 1
    start_time=time()
    
    while (sqrt(dot(target-x,target-x))>epsilon)
        x=update_crm2!(C,x,A,n,b,iter)
        iter += 1
    end
    iter-=1
    total_time=time()-start_time
    return x, iter,total_time
end


function solve_crm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, env::Gurobi.Env, ϵ::Real)
    x = copy(x0)
    m, n = size(y)
    iter = 0
    z = repeat(x, m)
    violation = []
    p1 = plot(aspect_ratio=1)
    if n == 2
        plot_all_elipsoids(p1, y, Q, k)
        plot_1ball(p1)
        scatter!(p1, [x0[1]], [x0[2]],
            markershape=:circle,
            label="x0",
            color=:orange,
            markersize=5)
    end
    points_x = []
    points_y = []
    while iter < max_iter
        update_crm!(z, y, Q, k, env)
        iter += 1
        Y = reshape(z, n, m)
        current_violation = -Inf
        current_violation_index = -1
        for i in 1:m
            violation_i = max_violation(Y[:, i], y, Q, k, ϵ)
            if violation_i > current_violation
                current_violation = violation_i
                current_violation_index = i
            end
        end
        push!(points_x, Y[1, current_violation_index])
        push!(points_y, Y[2, current_violation_index])
        # @show current_violation
        push!(violation, current_violation)
        if current_violation <= 0.0
            break
        end
    end
    if n == 2
        scatter!(p1, points_x, points_y,
            markershape=:circle,
            label="x_crm",
            color=:blue,
            markersize=3)
        scatter!(p1, [points_x[end]], [points_y[end]],
            markershape=:circle,
            label="x_crm_final",
            color=:red,
            markersize=3)
        legend_str = iter == 1 ? "iteration" : "iterations"
        title!(p1, "CRM method ($iter $legend_str)")
        savefig(p1, "projections_crm.pdf")
    end
    println("iter crm $iter")

    return z, violation
end

function test_cutting_plane(n::UInt, m::UInt, max_iter::UInt)
    env = Gurobi.Env()
    y = zeros(m, n)
    Q = zeros(m, n, n)
    k = zeros(m)
    Random.seed!(0)

    for i in 1:m
        # U, _, V = svd(randn(n, n))
        # s = LinRange(1.0, 1.00001, n)  # espectro entre 1 e 10
        # A = U * Diagonal(s) * V'
        # q = A * A'
        A = randn(n, n)
        q = A * A'
        # q = Diagonal(50 * rand(n))
        lambda_i = 10.0
        q += lambda_i * I(n)
        if !isposdef(q)
            println("ERROR: $q is not positive definite")
        end

        # if abs(maximum(vals)) / abs(minimum(vals)) > 1000
        #     println("ERROR: bad condition at $i")
        # end
        Q[i, :, :] = q
        y[i, :] .= rand(Float64, n) .* 20 .- 10
        k[i] = (1 + norm(y[i, :], 2)) * sqrt(norm(q, 2))
    end

    if n == 2
        for i in 1:m
            println("Q_$i = $(Q[i,:,:])")
        end
        for i in 1:m
            println("y_$i = $(y[i,:])")
        end

        println("k = ", k)
        p1 = plot(aspect_ratio=:equal)
        plot_all_elipsoids(p1, y, Q, k)
    end
    # println("k = ", k)
    x0 = zeros(n)
    x0[1] = 30
    x0[2] = 20
    ϵ = 1e-8

    target=projection_intersection(x0,p,Q,k,env)

    println("violation x0 = ", max_violation(x0, y, Q, k, ϵ))
    if n == 2
        println("x0 = ", x0)
    end

    if n == 2
        plot_1ball(p1)
        scatter!(p1, [x0[1]], [x0[2]],
            markershape=:circle,
            label="x0",
            color=:orange,
            markersize=6)
    end


    p2 = plot()

    @time x_3pm, violation_3pm = solve_a3pm_halpern(x0, y, Q, k, max_iter, env, ϵ)
    @show violation_3pm[end]
    plot!(p2, 1:length(violation_3pm), violation_3pm, label="3PM", lw=2)
    if n == 2
        scatter!(p1, [x_3pm[1]], [x_3pm[2]],
            markershape=:circle,
            label="x_3pm",
            color=:blue,
            markersize=6)
    end

    @time x_a3pm, violation_a3pm = solve_a3pm(x0, y, Q, k, max_iter, ϵ)
    @show violation_a3pm[end]
    # plot!(p2, 1:length(violation_a3pm), violation_a3pm, label="A3PM", lw=2)
    if n == 2
        scatter!(p1, [x_a3pm[1]], [x_a3pm[2]],
            markershape=:circle,
            label="x_a3pm",
            color=:red,
            markersize=7)
    end

    @time x_alt_proj, violation_alt = solve_alt_proj(x0, y, Q, k, max_iter, env, ϵ)
    @show violation_alt[end]
    plot!(p2, 1:length(violation_alt), violation_alt, label="alt_proj", lw=2)
    if n == 2
        scatter!(p1, [x_alt_proj[1]], [x_alt_proj[2]],
            markershape=:circle,
            label="x_alt_proj",
            color=:green,
            markersize=6)
    end

    @time x_cimmino, violation_cimmino = solve_cimmino(x0, y, Q, k, max_iter, env, ϵ)
    plot!(p2, 1:length(violation_cimmino), violation_cimmino, label="cimmino", lw=2)
    @show violation_cimmino[end]
    if n == 2
        scatter!(p1, [x_cimmino[1]], [x_cimmino[2]],
            markershape=:circle,
            label="x_cimmino",
            color=:black,
            markersize=6)
    end

    @time x_sccrm, violation_sccrm = solve_sc_crm(x0, y, Q, k, max_iter, env, ϵ)
    # plot!(p2, 1:length(violation_sccrm), violation_sccrm, label="sccrm", lw=2)
    @show violation_sccrm[end]
    if n == 2
        scatter!(p1, [x_sccrm[1]], [x_sccrm[2]],
            markershape=:circle,
            label="x_sccrm",
            color=:black,
            markersize=6)
    end


    @time x_crm, violation_crm = solve_crm(x0, y, Q, k, max_iter, env, ϵ)
    plot!(p2, 1:length(violation_crm), violation_crm, label="crm", lw=2)
    @show violation_crm[end]
    if n == 2
        Y = reshape(x_crm, Int(n), Int(m))
        max_violation_index = -1
        max_violation_val = -Inf
        for i in 1:m
            violation = max_violation(Y[:, i], y, Q, k, ϵ)
            if violation > max_violation_val
                max_violation_val = violation
                max_violation_index = i
            end
        end
        x_crm = Y[:, max_violation_index]
        scatter!(p1, [x_crm[1]], [x_crm[2]],
            markershape=:circle,
            label="x_crm",
            color=:purple,
            markersize=5)
        savefig(p1, "projections.pdf")
    end


    savefig(p2, "violations.pdf")
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

    # Plot
    p2 = plot()
    scatter!(p2, x[:, 1], x[:, 2], label="Points", legend=:topright, ms=8, markerstrokewidth=0)
    scatter!(p2, [c[1]], [c[2]], label="Circuncenter", ms=10, marker=:star5, color=:red)
    # Também desenhar linhas ligando os pontos
    plot!(p2, [x[:, 1]; x[1, 1]], [x[:, 2]; x[1, 2]], label="", lw=1, ls=:dash, color=:gray)

    title!(p2, "Circuncenter")
    xlabel!(p2, "x₁")
    ylabel!(p2, "x₂")
    savefig(p2, "circuncenter.pdf")
end

function test_corr_matrix(C::AbstractMatrix,n::Int,max_iter::Int,epsilon::Float64)
A=zeros(n,n,n)
b=zeros(n)
for i in 1:n 
     A[i,i,i]=1
     b[i]=1
end
target=zeros(n,n)

target,time_dijkstra=dijkstra(C,n,A,2,b,max_iter)
println("Time Dijkstra: $time_dijkstra")

newtarget=zeros(n,n)

  Xs,time_halpern=solve_alt_proj_halpern(C,A,b,n,max_iter,epsilon,target)
  println("Time alt proj: $time_halpern")

 z,iter,timecrm=solve_crm2(C,A,b,n,epsilon,target)
 println("Time CRM: $timecrm")

# return time_dijkstra,Xs,time_halpern,target
return time_dijkstra,timecrm,time_halpern


end


function main()
    n::Int= 100
    max_iter::Int= 200
    Aux = randn(n, n)
    C = Aux * Aux'
    epsilon=5.0
    #test_cutting_plane(n, m, max_iter)
    time_dijkstra,timecrm,time_halpern=test_corr_matrix(C,n,max_iter,epsilon)
    println("Time dijkstra: $time_dijkstra")
    println("Time CRM: $timecrm")  
    # m=length(Xs)
    # p = plot()
    # x=1:m
    # y=zeros(m)
    # for k in 1:m
    #     y[k]=sqrt(dot(Xs[k]-target,Xs[k]-target))
    #     println("$(y[k])") 
    # end
    # xlabel!(p, "Iterations")
    # # ylabel!(p, L"\sqrt{k}\,\|X_k - X_*\|")
    # plot!(p, x, y, label="", ls=:dash, color=:black)
    # savefig(p, "halpernsdp.pdf")
    # # sleep(1)
    # test_find_circuncenter(m, n)
end

main()