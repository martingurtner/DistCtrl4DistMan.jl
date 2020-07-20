"""
SquaredMatrixPlusIModule (C) 2020, Martin Gurtner
Efficient implementation of algebraic operations on matrices in the form D*D'+λ*I
"""
module SquaredMatrixPlusIModule

using LinearAlgebra

export SquaredMatrixPlusI

mutable struct SquaredMatrixPlusI{T<:Real} <: AbstractArray{T,2}
    # A = D*D' + λI
    D::AbstractArray{T};
    λ::T;
    tmp::Vector{T};

    function SquaredMatrixPlusI(D::AbstractArray{T}, λ::T;) where {T<:Real}
        tmp_size = length(size(D)) == 1 ? 1 : size(D)[2];
        new{T}(D, λ, zeros(T, tmp_size));
    end
end

function Base.size(sqm::SquaredMatrixPlusI)
    (size(sqm.D)[1], size(sqm.D)[1])
end

function LinearAlgebra.mul!(c::Vector{T}, A::SquaredMatrixPlusI{T}, b::Vector{T}) where T<:Real
    # c = (A.D*A.D' + A.λ*I)*b
    mul!(A.tmp, A.D', b);
    mul!(c, A.D, A.tmp);
    c .+= A.λ.*b;
end

end
