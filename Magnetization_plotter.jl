using CSV
using DataFrames
using Plots
using LaTeXStrings



filename = "MostStabilizeMFanswerT=0.0.csv"
df = CSV.read(filename, DataFrame)

df_sub = df[1:end, :]
h = df_sub[:, 1]
S1245x = df_sub[:, 2]
S1245y = df_sub[:, 3]
S1245z = df_sub[:, 4]
S1346x = df_sub[:, 5]
S1346y = df_sub[:, 6]
S1346z = df_sub[:, 7]
S2356x = df_sub[:, 8]
S2356y = df_sub[:, 9]
S2356z = df_sub[:, 10]
SAx = df_sub[:, 11]
SAy = df_sub[:, 12]
SAz = df_sub[:, 13]
SBx = df_sub[:, 14]
SBy = df_sub[:, 15]
SBz = df_sub[:, 16]
SCx = df_sub[:, 17]
SCy = df_sub[:, 18]
SCz = df_sub[:, 19]

S_hexiagon_x=(S1245x+S1346x+S2356x)./2 #6角形上のスピンの合計についての平均
S_hexiagon_y=(S1245y+S1346y+S2356y)./2
S_hexiagon_z=(S1245z+S1346z+S2356z)./2
S_totz=(S_hexiagon_z.+SAz.+SBz.+SCz) #全体のz成分の平均

p = plot(
    h, 
    [S_hexiagon_x, S_hexiagon_y, S_hexiagon_z, SAx, SAy, SAz, S_totz], # データを配列としてまとめる
    label = [L"S_{hex,x}" L"S_{hex,y}" L"S_{hex,z}" L"S_{A,x}" L"S_{A,y}" L"S_{A,z}" L"S_{tot,z}"],
    color = [:blue :red :green :cyan :magenta :orange :black], # 系列ごとの色
    shape = [:circle :rect :diamond :utriangle :dtriangle :hexagon :cross], # 系列ごとのマーカー図形
    markersize = 1,           # マーカーの大きさ
    markerstrokewidth = 0,    # マーカーの枠線を消す
    every=100,
    lw = 2,                   # 線の太さ
    xlabel = L"External Magnetic Field $h$",
    ylabel = "Magnetization", # 縦軸の名前
    framestyle = :box,
    grid = false,
    legend = :outerright,     # レジェンドをグラフの外側に配置（重なり防止）
    dpi = 300                 # 高解像度
)

dir = "results"

filename = joinpath(dir,"magnetizationT=0.0_RevicedMay20.pdf")

# 保存
savefig(p, filename)