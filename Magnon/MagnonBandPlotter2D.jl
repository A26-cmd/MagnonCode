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
using Plots
using LaTeXStrings
using OffsetArrays

MF=[8.09275719708984e-17,-1.004929098203616e-16,4.000000000000003,3.257444709950226e-16,-7.62028885373558e-17,4.000000000000003,-1.1460247777200346e-16,-3.715632614594998e-16,4.000000000000003,-1.2801588640098996e-17,2.2542854248859202e-17,1.0000000000000007,-1.721687328488837e-16,8.401277571960353e-17,1.0000000000000007,-1.6983957364290833e-16,1.260595634998718e-17,1.0000000000000007
]
MF = OffsetArray(MF, 0:length(MF)-1)
const J=1.0
const h=7.578947368421052

#バンドプロット　経路はSchwinger Bosonのままなので一旦意味ない結果
# バンドの数を選べる。基底状態に近い順から。最大66バンド
N_bands = 6
ω = [Float64[] for _ in 1:N_bands]

Ns = 50
Gamma = [0.0, 0.0]
M_point = [0.0, sqrt(3)/2]
K_point = [0.5, sqrt(3)/2]
path1 = [(Gamma[1] + (M_point[1] - Gamma[1]) * (i / Ns), Gamma[2] + (M_point[2] - Gamma[2]) * (i / Ns)) for i in 0:Ns-1]
path2 = [(M_point[1] + (K_point[1] - M_point[1]) * (i / Ns), M_point[2] + (K_point[2] - M_point[2]) * (i / Ns)) for i in 0:Ns-1]
path3 = [(K_point[1] + (Gamma[1] - K_point[1]) * (i / Ns), K_point[2] + (Gamma[2] - K_point[2]) * (i / Ns)) for i in 0:Ns] # 最後まで含む
paths = [path1, path2, path3]

for path in paths
    for (qx, qy) in path
        k = [qx, qy]
        val = Magnonbandmodule.disp(J, h, MF, k)
        for b in 1:N_bands
            push!(ω[b], val[b])
        end
    end
end

# 各区間の物理的な道のり（距離）
L1 = sqrt(3)/2  # Gamma -> M
L2 = 0.5         # M -> K
L3 = 1.0         # K -> Gamma

# 横軸の累積パラメータ（s_list）を 3区間用に正しく累積定義
s1_plot = range(0.0, L1, Ns+1)[1:end-1]
s2_plot = range(L1, L1 + L2, Ns+1)[1:end-1]
s3_plot = range(L1 + L2, L1 + L2 + L3, Ns+1)[1:end]
s_list = vcat(s1_plot, s2_plot, s3_plot)

# 全データの中から最小値と最大値を自動取得
all_values = vcat(ω...)
y_min = minimum(all_values)
y_max = maximum(all_values)
y_pad = (y_max - y_min) * 0.05

# 縦軸の範囲を自動決定
ylim_bottom = y_min > -1e-5 ? 0.0 : y_min - y_pad
ylim_top = y_max + y_pad

# プロット初期化
p = plot(
    s_list, ω[1],
    xlim = (0.0, s_list[end]),
    ylim = (ylim_bottom, ylim_top),
    lc = :blue,
    lw = 1.5,
    xlabel = "",
    ylabel = L"\omega_{\mathbf{k}}",
    framestyle = :box,
    grid = false,
    guidefontsize = 18,
    tickfontsize = 14,
    margin = Plots.Measures.Length(:mm, 8.0),
    legend = false,
)

# 残りのバンドを重ね書き
for b in 2:N_bands
    plot!(s_list, ω[b], lw = 1.5, lc = :blue)
end

# --- 【修正】3区間の高対称点に合わせて目盛りを直接マッピング ---
xticks_pos = [0.0, L1, L1 + L2, L1 + L2 + L3]
xticks_line = [L1, L1 + L2]
xticks_lbl = [L"\Gamma", L"\mathrm{M}", L"\mathrm{K}", L"\Gamma"]
xticks!(xticks_pos, xticks_lbl)

# 高対称線の描画（黒い点線でセクションを区切る）
vline!(xticks_line, c=:black, ls=:dash, lw=0.8, label=false)

# 保存
savefig(p, "testdispersion_m=9over9test2.png")