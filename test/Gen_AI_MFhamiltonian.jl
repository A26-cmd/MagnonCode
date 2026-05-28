


using StaticArrays
using LinearAlgebra
using Roots
using DataFrames
using CSV
using SparseArrays

function get_spin_operators(n_sites::Int)
    dim = 2^n_sites
    Sx_list = []
    Sy_list = []
    Sz_list = []

    for i in 1:n_sites
        shift = i - 1  # ビット位置
        
        # --- Sz の構築 (対角行列) ---
        diag_z = [((s >> shift) & 1 == 1 ? 0.5 : -0.5) for s in 0:(dim-1)]
        push!(Sz_list, sparse(1:dim, 1:dim, diag_z, dim, dim))

        # --- S+ と S- を構築して Sx, Sy を作る ---
        # S+ : |0> -> |1>, S- : |1> -> |0>
        I_p, J_p, V_p = Int[], Int[], Float64[]
        I_m, J_m, V_m = Int[], Int[], Float64[]

        for s in 0:(dim-1)
            if ((s >> shift) & 1) == 0 # i番目が0ならS+が可能
                push!(I_p, (s | (1 << shift)) + 1); push!(J_p, s + 1); push!(V_p, 1.0)
            else # i番目が1ならS-が可能
                push!(I_m, (s & ~(1 << shift)) + 1); push!(J_m, s + 1); push!(V_m, 1.0)
            end
        end
        Sp = sparse(I_p, J_p, V_p, dim, dim)
        Sm = sparse(I_m, J_m, V_m, dim, dim)

        push!(Sx_list, 0.5 * (Sp + Sm))
        push!(Sy_list, -0.5im * (Sp - Sm))
    end

    return Sx_list, Sy_list, Sz_list
end

function build_and_solve_mf_hamiltonian(J::Float64, h::Float64, m_vals::Vector{Float64})
    N = 9
    dim = 2^N
    Sx, Sy, Sz = get_spin_operators(N)

    # サイト番号の定義 (1-based)
    # 1-6: 六角形, 7:A, 8:B, 9:C
    A, B, C = 7, 8, 9

    # 1. 六角形内部の Heisenberg 項 (Hc)
    # Hc = J * Σ (Si ⋅ Sj) = J * Σ (SixSjx + SiySjy + SizSjz)
    H_MF = spzeros(ComplexF64, dim, dim)
    hex_edges = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)]
    for (i, j) in hex_edges
        H_MF += J * (Sx[i]*Sx[j] + Sy[i]*Sy[j] + Sz[i]*Sz[j])
    end

    # 2. 平均場項 (画像の赤枠内の式)
    # ※ 一般的な期待値は z 方向と仮定して Sz 演算子を使用します
    
    # 副格子 A 関連: <S1+S2+S4+S5>SA + (S1+S2+S4+S5)<SA>
    m_sum_A = m_vals[1] + m_vals[2] + m_vals[4] + m_vals[5]
    H_MF += J * (m_sum_A * Sz[A] + (Sz[1] + Sz[2] + Sz[4] + Sz[5]) * m_vals[A])

    # 副格子 B 関連: <S1+S3+S4+S6>SB + (S1+S3+S4+S6)<SB>
    m_sum_B = m_vals[1] + m_vals[3] + m_vals[4] + m_vals[6]
    H_MF += J * (m_sum_B * Sz[B] + (Sz[1] + Sz[3] + Sz[4] + Sz[6]) * m_vals[B])

    # 副格子 C 関連: <S2+S3+S5+S6>SC + (S2+S3+S5+S6)<SC>
    m_sum_C = m_vals[2] + m_vals[3] + m_vals[5] + m_vals[6]
    H_MF += J * (m_sum_C * Sz[C] + (Sz[2] + Sz[3] + Sz[5] + Sz[6]) * m_vals[C])

    # 3. ゼーマン項: -h * Σ Siz
    for i in 1:N
        H_MF -= h * Sz[i]
    end

    # 行列が小さい(512x512)ので密行列にして対角化
    H_dense = Matrix(H_MF)
    res = eigen(H_dense)

    return H_MF, res.values, res.vectors
end

# === 使用例 ===
J = 1.0
h = 0.5
# 各サイトの期待値 <Sz> の初期仮定 (例: すべて 0.2)
m_initial = fill(0.2, 9)

H, evals, evecs = build_and_solve_mf_hamiltonian(J, h, m_initial)

println("基底状態エネルギー: ", minimum(real.(evals)))
println("ハミルトニアンのサイズ: ", size(H))