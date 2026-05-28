module CppIterationModule

using StaticArrays
using LinearAlgebra
using Roots
using DataFrames
using CSV
using SparseArrays


using LinearAlgebra
#今スピンをパウリ行列そのままで表現している。

#平均場ハミルトニアンを解く
#MFの与え方：<S1245>_x,<S_1245>_y,<S1245>_z,<S1346>_x,<S1346>_y,<S1346>_z,<S2356>_x,<S2356>_y,<S2356>_z,<SA>_x,<SA>_y, <SA>_z,<SB>_x,<SB>_y,<SB>_z,<SC>_x,<SC>_y,<SC>_zの順
#c1,c2:0~5サイトは研究ノートの通り、6,7,8がA,B,C
#MFを基底状態に対して計算してくれる。
function solve_mf_physics(J::Float64, h::Float64, m_fields::Vector{Float64})
    ndim = 512
    # 固有値(512) + 固有ベクトル(512*512) を格納する複素数配列
    out_buffer = Vector{ComplexF64}(undef, ndim + ndim^2)
    MFout = Vector{Float64}(undef, 18)

    ccall((:build_and_solve_mf_hamiltonian, "./lib_make_hamiltonian.so"), Cvoid,
          (Float64, Float64, Ptr{Float64},Ptr{Float64}, Ptr{ComplexF64}),
          J, h, m_fields,MFout, out_buffer)
    # 最初の512個は固有値（実数部だけ取り出す）
    evals = real.(out_buffer[1:ndim])
    
    # 残りを 512x512 の行列に整形
    evecs = reshape(out_buffer[ndim+1:end], ndim, ndim)

    #平均場を出力

    return evals, evecs, MFout
end



function iteration(J::Float64, h::Float64, m_init::Vector{Float64}; N_iter=10000, eps=1e-8, α_val=0.4)
    
    
    m_current = copy(m_init)

    evals = nothing
    evecs = nothing

    for i in 1:N_iter
        evals, evecs, m_new = solve_mf_physics(J, h, m_current)
        if i % 100 == 0
            println("Iteration $i: Mean fields = ", m_new)
        end
        if all(abs.(m_new .- m_current) .< eps)
            println("--- All components converged! ---")
            return evals, evecs, m_new
        end

        # 3. 線形混合 (Linear Mixing)
        # 次のステップの入力を、古い値(m_current)と新しい値(m_new)のブレンドにする
        # m_current = α * 古い値 + (1 - α) * 新しい値
        m_current = α_val .* m_current .+ (1.0 - α_val) .* m_new
    end

    println("Warning: Did not converge within $N_iter iterations.")
    return evals, evecs, m_current
end

J=1.0
h=0.0
m_current = zeros(Float64, 18)


m_current[2]=4.0

m_current[5]=4.0


m_current[8]=4.0

m_current[11]=1.0

m_current[14]=1.0

m_current[17]=1.0
evals, evecs, m_final = iteration(J/4, h, m_current)#正しくスピンの値を表現するにはJを4でわる必要がある
println("Final mean fields: ", m_final)


end # module