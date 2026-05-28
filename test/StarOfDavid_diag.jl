using SparseArrays
using LinearAlgebra

"""
    build_star_of_david_hamiltonian(J::Float64, h::Float64)

ダビデの星(12サイト)上のS=1/2 Heisenberg模型 + 外部磁場(Z方向)のハミルトニアンを構築する。
基底は0から2^12-1までの整数（ビット表示）を用いる。
"""
function build_star_of_david_hamiltonian(J::Float64, h::Float64)
    N = 12
    dim = 2^N

    # 疎行列用のインデックスと値の配列
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]

    # ダビデの星の結合(エッジ)を定義
    # ビットシフトの都合上、サイト番号を 0 ~ 11 で定義します
    # 0~5: 内部の六角形, 6~11: 外側の頂点
    edges = [
        # 内部の六角形
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,0),
        # 外側の頂点が六角形に付随する結合（三角形を形成）
        (0,6), (1,6),   # 頂点6 は 0, 1 に接続
        (1,7), (2,7),   # 頂点7 は 1, 2 に接続
        (2,8), (3,8),   # 頂点8 は 2, 3 に接続
        (3,9), (4,9),   # 頂点9 は 3, 4 に接続
        (4,10), (5,10), # 頂点10 は 4, 5 に接続
        (5,11), (0,11)  # 頂点11 は 5, 0 に接続
    ]

    # すべての基底状態(000...0 から 111...1)に対してループ
    for s in 0:(dim-1)
        diag_val = 0.0

        # --- 1. ゼーマン項 (外部磁場 h) ---
        # -h ∑ S_i^z
        for i in 0:(N-1)
            bit_i = (s >> i) & 1
            sz = bit_i == 1 ? 0.5 : -0.5
            diag_val -= h * sz
        end

        # --- 2. ハイゼンベルグ相互作用項 (J) ---
        # J ∑ (S_i^z S_j^z + 1/2 (S_i^+ S_j^- + S_i^- S_j^+))
        for (i, j) in edges
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1

            if bit_i == bit_j
                # スピンが平行な場合 (↑↑ または ↓↓): S_i^z S_j^z = 0.25
                diag_val += J * 0.25
            else
                # スピンが反平行な場合 (↑↓ または ↓↑): S_i^z S_j^z = -0.25
                diag_val -= J * 0.25

                # 非対角項 (スピンフリップ): J/2 * (S_i^+ S_j^- + S_i^- S_j^+)
                # XOR(⊻)を使って i番目 と j番目 のビットを反転させる
                s_prime = s ⊻ ((1 << i) | (1 << j))

                # Juliaは1-based indexなので s+1, s_prime+1 にする
                push!(I_idx, s + 1)
                push!(J_idx, s_prime + 1)
                push!(V, J * 0.5)
            end
        end

        # 対角成分の追加
        push!(I_idx, s + 1)
        push!(J_idx, s + 1)
        push!(V, diag_val)
    end

    # 疎行列を生成して返す
    return sparse(I_idx, J_idx, V, dim, dim)
end

# === 実行例 ===
J_val = 1.0  # 反強磁性ハイゼンベルグ相互作用
h_val = 0.5  # 外部磁場

println("ハミルトニアンを構築中...")
H_sparse = build_star_of_david_hamiltonian(J_val, h_val)

println("基底状態のエネルギーを計算中 (次元: $(size(H_sparse, 1)) x $(size(H_sparse, 2)))...")
# 4096 x 4096程度なら密行列に変換して直接対角化しても数秒で終わります
H_dense = Matrix(H_sparse)
energies = eigvals(H_dense)

println("基底状態エネルギー: E_0 = $(energies[1])")