include("Bogoliubov.jl")
include("Magnon.jl")

using .Magnonbandmodule
using .BogoliubovModule
using LinearAlgebra
using Match
using SparseArrays
using StaticArrays
using Roots
using DataFrames
using CSV
using PlotlyJS  # Plots から PlotlyJS へ変更
using LaTeXStrings
using OffsetArrays

# --- データの準備 ---
MF=[-1.2321516463085455e-16,-2.9879101127586496e-16,-1.3333333333333326,-1.8855240701474897e-16,-4.124222032811007e-16,-1.3333333333333335,3.7585069004161183e-16,-5.083582851751754e-16,-1.3333333333333341,-2.581382827465801e-17,3.055550010507352e-16,1.0,1.7425126949271068e-16,2.177210216024491e-17,1.0,-2.1543342048678105e-16,5.486701083064956e-16,1.0
]
MF = OffsetArray(MF, 0:length(MF)-1)
const J = 1.0
const h=0.2706766917293233

N_bands = 7

# --- ① ブリルアンゾーン（BZ）全体のグリッド生成 ---
# K点（0.5, sqrt(3)/2）の物理的対称性から、2次元面内のプロット範囲を決定
nk = 100
qx_vals = range(-1.2, 1.2, length=nk)
qy_vals = range(-1.2, 1.2, length=nk)

# 各バンドのエネルギーを格納する3次元配列
Z_bands = zeros(Float64, nk, nk, N_bands)

# 外側のループをマルチスレッド化します
# 各スレッドに異なる qx (ix) の行が割り振られます
Threads.@threads for ix in 1:nk
    qx = qx_vals[ix]
    
    # 内側のループは通常通り回します
    for iy in 1:nk
        qy = qy_vals[iy]
        k = [qx, qy]
        try
            # 固有値（バンドエネルギー）の計算
            val = Magnonbandmodule.disp(J, h, MF, k)
            
            for b in 1:N_bands
                # 割り当てられたメモリ位置 [ix, iy, b] に書き込むため、スレッド間での衝突はありません
                Z_bands[ix, iy, b] = val[b]
            end
        catch e 
            if e isa LinearAlgebra.PosDefException || (e isa TaskFailedException && occursin("PosDef", string(e)))
                println("Non-positive definite matrix skipped at k = [$qx, $qy]")
            end
            for b in 1:N_bands
                Z_bands[ix, iy, b] = NaN  # または NaN
            end
        end

    end
end

# 全エネルギーから最大値・最小値を取得（軸の範囲の自動決定用）
z_min = minimum(Z_bands)
z_max = maximum(Z_bands)
z_pad = (z_max - z_min) * 0.05

# --- ② 高対称点（ラベル）の3次元空間への配置 ---
Gamma = [0.0, 0.0]
M_point = [0.0, sqrt(3)/2]
K_point = [0.5, sqrt(3)/2]

# プロット上に浮かび上がらせる高対称点のリスト
# Z座標は、見やすさのために「バンドの底（z_min）」の位置に固定します
points_x = [Gamma[1], M_point[1], K_point[1]]
points_y = [Gamma[2], M_point[2], K_point[2]]
points_z = [z_min, z_min, z_min]
points_text = ["Γ (0, 0)", "M (0, √3/2)", "K (0.5, √3/2)"]

# --- ③ PlotlyJS オブジェクトの構築 ---
traces = GenericTrace[]

# 各バンドを面（Surface）として追加
for b in 1:N_bands
    push!(traces, surface(
        x=qx_vals, y=qy_vals, z=Z_bands[:, :, b],
        colorscale="Viridis",
        showscale= (b == 1), # カラーバーは1つだけ表示
        name="Band $b",
        opacity=0.85
    ))
end

# 高対称点のインデックス（マーカーとテキスト）を追加
push!(traces, scatter3d(
    x=points_x, y=points_y, z=points_z,
    mode="markers+text",
    marker=attr(size=6, color="red", symbol="circle"),
    text=points_text,
    textposition="top center",
    textfont=attr(size=14, color="black"),
    name="High-symmetric Points"
))

# レイアウトの設定（最大値・最小値を自動反映）
layout = Layout(
    title="3D Magnon Band Structure over BZ",
    autosize=true,
    width=900, height=750,
    scene=attr(
        xaxis=attr(title="qx", range=[-1.2, 1.2]),
        yaxis=attr(title="qy", range=[-1.2, 1.2]),
        zaxis=attr(title="Energy ω_k", range=[z_min - z_pad, z_max + z_pad]),
        camera=attr(eye=attr(x=1.5, y=1.5, z=1.2)) # 初期のカメラ角度
    ),
    margin=attr(l=0, r=0, b=0, t=40)
)

# プロットの表示・保存
p_3d = plot(traces, layout)

# HTML形式で保存（ブラウザで開けばいつでもグリグリ動かせます）
PlotlyJS.savefig(p_3d, "3d_magnon_bandm=1over9T=0.html")
display(p_3d)