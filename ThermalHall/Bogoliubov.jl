module BogoliubovModule #モジュールの作成

using LinearAlgebra

function Bogoliubov(matrix)
    N_mat = size(matrix,1)
    Kd = cholesky(matrix).L
    K = adjoint(Kd)
    τ = ones(Float64,N_mat)
    τ[div(N_mat,2)+1:end] .*=-1
    w,v = eigen(K*Diagonal(τ)*Kd)
    index = sortperm(real(w),rev = true)
    w = w[index]
    v = v[:,index]
    U = inv(K)*v*Diagonal(sqrt.(w .*τ))
    return w,U
end

end