module LinSolvers

using LinearAlgebra

export kernel_solve, kernel_solve!, LinSystemData

struct LinSystemData{T<:Real, I<:Int}
    n::I;
    m::I;
    A::Matrix{T};
    b::Vector{T};
    R::Matrix{T};
    x_sol::Vector{T};

    LinSystemData{T}(n::I, m::I) where {I<:Int, T<:Real} = new{T, I}(
            n, m,
            zeros(T, m+n,m),                   # A
            zeros(T, m),                       # b
            zeros(T, m,m),    # R
            zeros(T, n)
    );

    LinSystemData(n::I, m::I, x_sol::Vector{T}) where {I<:Int, T<:Real} = new{T, I}(
        n, m,
        zeros(T, m+n,m),  # A
        zeros(T, m),      # b
        zeros(T, m,m),    # R
        x_sol
    );
end # struct

"""
    update!(lsd::LinSystemData, Dᵀ::Matrix, λ<:Real, b::Vector)

Update 'lsd' structure storing data for solving matrix equation (DᵀD + 1/λI)x=Dᵀb.
"""
function update!(lsd::LinSystemData{T}, Dᵀ::Matrix{T}, λ::T, b::Vector{T}) where T<:Real
    @assert (size(Dᵀ, 1) == lsd.n) && (size(Dᵀ, 2) == lsd.m) "LinSystemData does not have compatible dimensions with matrix Dᵀ."
    @assert length(b) == lsd.m "LinSystemData does not have compatible dimensions with vector b."

    @inbounds for i in 1:lsd.n, j in 1:lsd.m
        lsd.A[i,j] = Dᵀ[i,j];
    end

    diag_val = 1/sqrt(λ);
    @inbounds for i in 1:lsd.m, j in 1:lsd.m
        lsd.A[i+lsd.n,j] = i==j ? diag_val : 0;
    end

    copyto!(lsd.b, b);

    nothing
end


"""
    kernel_solve(Dᵀ::Matrix, λ<:Real, b::Vector)

Solve a matrix equation in form (DᵀD + 1/λ)x=Dᵀb.

# Examples
```julia-repl
julia> n, m = 10, 3;
julia> for i in 1:10
           x_sol = kernel_solve(rand(n,m), 1.0, rand(m));
           print("Solution is \$x_sol\\n");
       end
Solution is ...
```
"""
function kernel_solve(Dt::Matrix{T}, λ::T, b::Vector{T}) where T<:Real
    n, m = size(Dt);
    A = zeros(m+n, m);
    b_tmp = zeros(m);
    x_sol = zeros(n);
    R = zeros(m, m);

    A[1:n,1:m] = Dt;
    @simd for i in (n+1):(n+m)
        A[i,i-n] = 1/sqrt(λ);
    end

    qr_r!(A, R);

    copyto!(b_tmp, b);
    forward_substitution!(R, b_tmp)
    backward_substitution!(R, b_tmp)

    mul!(x_sol, Dt, b_tmp);
    return x_sol;
end

"""
    kernel_solve!(lsd::LinSystemData, Dᵀ::Matrix, λ<:Real, b::Vector)

Solve a matrix equation in form (DᵀD + 1/λ)x=Dᵀb using the pre-allocated data structure 'lsd'.

# Example
```julia-repl
julia> n, m = 10, 3;
julia> lsd = LinSystemData{Float64}(n, m);
julia> for i in 1:10
           kernel_solve!(lsd, rand(n,m), 1.0, rand(m));
           print("Solution is \$(lsd.x_sol)\\n");
       end
Solution is ...
```
"""
function kernel_solve!(lsd::LinSystemData{T}, Dt::Matrix{T}, λ::T, b::Vector{T}) where T<:Real
    update!(lsd, Dt, λ, b);

    qr_r!(lsd.A, lsd.R);

    forward_substitution!(lsd.R, lsd.b);
    backward_substitution!(lsd.R, lsd.b);

    mul!(lsd.x_sol, Dt, lsd.b);

    nothing
end

"""
    kernel_solve_prealloc!(Dt::Matrix{T}, λ::T, Fi::Vector{T}, x_sol::Vector{T},
            A::Matrix{T},
            b_tmp::Vector{T},
            R::Matrix{T}) where T<:Real

Solve a matrix equation in form (DᵀD + 1/λ)x=Dᵀb with preallocated vectors and matrices provided.



# Examples
```julia-repl
julia> n, m = 10, 3;
julia> for i in 1:10
           x_sol = kernel_solve(rand(n,m), 1.0, rand(m));
           print("Solution is \$x_sol\\n");
       end
Solution is ...
```
"""
function kernel_solve_prealloc!(Dt::Matrix{T}, λ::T, b::Vector{T}, x_sol::Vector{T},
        A::Matrix{T},
        b_tmp::Vector{T},
        R::Matrix{T}) where T<:Real
    n, m = size(Dt);

    @inbounds for i in 1:n, j in 1:m
        A[i,j] = Dt[i,j];
    end

    diag_val = 1/sqrt(λ);
    @inbounds for i in 1:m, j in 1:m
        A[i+n,j] = i==j ? diag_val : 0;
    end

    qr_r!(A, R);

    copyto!(b_tmp, b);
    forward_substitution!(R, b_tmp)
    backward_substitution!(R, b_tmp)

    mul!(x_sol, Dt, b_tmp);

    nothing
end

# Forward substitution
function forward_substitution!(R::Matrix, b::Vector)
    n = size(R,1);
    @assert n==length(b) "R and b must have compatible dimensions!"

    @fastmath @inbounds for i in 1:n
        @simd for j in 1:(i-1)
            b[i] -= b[j]*R[j,i];
        end
        b[i] /= R[i,i];
    end

    nothing
end

# Backward substitution
function backward_substitution!(R::Matrix, b::Vector)
    n = size(R,1);
    @assert n==length(b) "R and b must have compatible dimensions!"

    @fastmath @inbounds for i in n:-1:1
        @simd for j in (i+1):n
            b[i] -= b[j]*R[i,j];
        end
        b[i] /= R[i,i];
    end

    nothing
end

function qr_r!(A::Matrix, R::Matrix)
    m, n = size(A);
    @assert n==size(R,1) "Dimensions of A and R must be compatible!"

    @fastmath @inbounds for i in 1:n
        R[i,i] = sqrt(colTimesCol(A, i, i));
        for j in range(i+1, stop=n)
            R[i,j] = colTimesCol(A, i , j)/R[i,i];
            α = R[i,j]/R[i,i];
            for k in 1:m
                A[k,j] -= A[k,i]*α;
            end
        end
    end

    nothing
end

function colTimesCol(A::Matrix, i::Int, j::Int)
    res = 0.0;
    n = size(A,1);
    @inbounds @simd for k in 1:n
        res += A[k,i]*A[k,j];
    end

    return res
end

end
