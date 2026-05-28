module Magnonbandmodule

export Partlal_Basis_hex
export Partial_basis_ABCsite
export Kernel_J
export cal_Kernel


include("Bogoliubov.jl")

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
#六角形中心とA ,B,Cサイトの相対位置 kagomeの最小単位胞a=1
const δA=[-3/4,sqrt(3)/4]
const δB=[3/4, sqrt(3)/4]
const δC =[0,sqrt(3)/2]

I = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
X = sparse([0.0 1.0+0.0im; 1.0+0.0im 0.0])
Y = sparse([0.0 -1.0im; 1.0im 0.0])
Z = sparse([1.0+0.0im 0.0; 0.0 -1.0+0.0im])

#サイトのインデックスは0スタート
function times(list,N=6)
    list = OffsetArray(list, 0:N-1)
    list[0] = kron(list[0], list[1])
    for i in 2:length(list)-1
        list[0] = kron(list[0], list[i])
    end
    return list[0]
end

function set_spins(sites, σs,N=6)
    list_mats = fill(I, N)
    list_mats = OffsetArray(list_mats, 0:N-1)
    for (site, σ) in zip(sites, σs)
        list_mats[site] = σ
    end
    return list_mats
end

function I_ndim(N=6)
    list = fill(I,N)
    return times(list)
end
function S_idotS_j_op(i,j,J)
    return J*times(set_spins([i,j],[X,X]))+J*times(set_spins([i,j],[Y,Y]))+J*times(set_spins([i,j],[Z,Z]))

end

function x_op(i,val,N=6)
    return val*times(set_spins([i],[X],N))
end

function y_op(i,val,N=6)
    return val*times(set_spins([i],[Y],N))
end

function z_op(i,val,N=6)
    return val*times(set_spins([i],[Z],N))
end

function Partlal_Basis_hex(J,h,MF)
    H = zeros(ComplexF64, 2^6, 2^6)
    #六角形部分
    for i in 0:4
        H += S_idotS_j_op(i,i+1,J)
    end
    H+= S_idotS_j_op(0,5,J)
    for i in 0:5
        H+= x_op(0,(MF[9]+MF[12])*J)+
            y_op(0,(MF[10]+MF[13])*J)+
            z_op(0,(MF[11]+MF[14])*J-h)+
            x_op(1,(MF[9]+MF[15])*J)+
            y_op(1,(MF[10]+MF[16])*J)+
            z_op(1,(MF[11]+MF[17])*J-h)+
            x_op(2,(MF[12]+MF[15])*J)+
            y_op(2,(MF[13]+MF[16])*J)+
            z_op(2,(MF[14]+MF[17])*J-h)+
            x_op(3,(MF[9]+MF[12])*J)+
            y_op(3,(MF[10]+MF[13])*J)+
            z_op(3,(MF[11]+MF[14])*J-h)+
            x_op(4,(MF[9]+MF[15])*J)+
            y_op(4,(MF[10]+MF[16])*J)+
            z_op(4,(MF[11]+MF[17])*J-h)+
            x_op(5,(MF[12]+MF[15])*J)+
            y_op(5,(MF[13]+MF[16])*J)+
            z_op(5,(MF[14]+MF[17])*J-h)
    end
    evals, evecs = eigen(H)
    return evals, evecs
end

#iはA,B,Cサイトのインデックス
function Partial_basis_ABCsite(J,h,MF,i)
    if i == "A" 
        j=0
    elseif i == "B"
        j=1
    elseif i == "C"
        j=2
    else
        error("unknown site")
    end
    H = zeros(ComplexF64, 2, 2)
    H += X*(MF[3*j+0]*J)+
        Y*(MF[3*j+1]*J)+
        Z*(MF[3*j+2]*J-h)
    evals, evecs = eigen(H)
    return evals, evecs
end
#exp(-imδA,B,C)\bar O^{hex}_{alpha n}を計算 MFと同じ名付け方
#δ(S1+S2+S4+S5)X=0　δ(S1+S2+S4+S5)Y=1 δ(S1+S2+S4+S5)Z=2 δ(S1+S3+S4+S6)X=3 δ(S1+S3+S4+S6)Y=4 δ(S1+S3+S4+S6)Z=5 δ(S2+S3+S5+S6)X=6 δ(S2+S3+S5+S6)Y=7 δ(S2+S3+S5+S6)Z=8  
function Hex_cal_matrix_elementForJAB(evecs,alpha,n,k)
    gs = evecs[:,1]
    operator = zeros(ComplexF64, 2^6, 2^6)
    operator = @match alpha begin
        0 => exp(im*dot(k,δA))*(x_op(0,1)+x_op(1,1))+exp(im*dot(k,-δA))*(x_op(3,1)+x_op(4,1))
        1 => exp(im*dot(k,δA))*(y_op(0,1)+y_op(1,1))+exp(im*dot(k,-δA))*(y_op(3,1)+y_op(4,1))
        2 => exp(im*dot(k,δA))*(z_op(0,1)+z_op(1,1))+exp(im*dot(k,-δA))*(z_op(3,1)+z_op(4,1))
        3 => exp(im*dot(k,δB))*(x_op(0,1)+x_op(5,1))+exp(im*dot(k,-δB))*(x_op(3,1)+x_op(2,1))
        4 => exp(im*dot(k,δB))*(y_op(0,1)+y_op(5,1))+exp(im*dot(k,-δB))*(y_op(3,1)+y_op(2,1))
        5 => exp(im*dot(k,δB))*(z_op(0,1)+z_op(5,1))+exp(im*dot(k,-δB))*(z_op(3,1)+z_op(2,1))
        6 => exp(im*dot(k,δC))*(x_op(1,1)+x_op(2,1))+exp(im*dot(k,-δC))*(x_op(4,1)+x_op(5,1))
        7 => exp(im*dot(k,δC))*(y_op(1,1)+y_op(2,1))+exp(im*dot(k,-δC))*(y_op(4,1)+y_op(5,1))
        8 => exp(im*dot(k,δC))*(z_op(1,1)+z_op(2,1))+exp(im*dot(k,-δC))*(z_op(4,1)+z_op(5,1))
    end
    return dot(evecs[:,n+1], operator*gs)
end

#A,B,Cの孤立サイトの励起状態はは一つしかない
function ABC_cal_matrix_element(evecs,xyz)
    operator = zeros(ComplexF64, 2, 2)
    operator = @match xyz begin
        0 => X
        1 => Y
        2 => Z
    end
    gs = evecs[:,1]
    return dot(evecs[ : ,2],operator*gs) #A,B,Cサイトの局所励起状態は一つしかない
end


#共鳴マグのんの位置を六角形中心において計算
#六角形とA,B,Cサイトの相互作用部分
#siteでA,B,Cを入力 angleでx,y,zを指定 那須先生の一般化HPの書類の定義に準拠,バーなし0,バーあり1
function Kernel_J(site,bar,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    Mat= zeros(ComplexF64,63,1)
    for n in 1:63
        if bar == 0
            if site == "A"
                for alpha in 0:2
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*ABC_cal_matrix_element(Aevecs,mod(alpha,3))
                end
            elseif site == "B"
                for alpha in 3:5
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*ABC_cal_matrix_element(Bevecs,mod(alpha,3))
                end
            else site == "C"
                for alpha in 6:8
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*ABC_cal_matrix_element(Cevecs,mod(alpha,3))
                end
            end
        else
            if site == "A"
                for alpha in 0:2
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*conj(ABC_cal_matrix_element(Aevecs,mod(alpha,3)))
                end
            elseif site == "B"
                for alpha in 3:5
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*conj(ABC_cal_matrix_element(Bevecs,mod(alpha,3)))
                end
            
            else site == "C"
                for alpha in 6:8
                    Mat[n,1] += Hex_cal_matrix_elementForJAB(Hevecs,alpha,n,k)*conj(ABC_cal_matrix_element(Cevecs,mod(alpha,3)))
                end
            end
        end
    end
    
    return J*Mat

end

function calΔE_hex(Hevals)
    ΔE = zeros(ComplexF64,63,63)
    for i in 1:63
        ΔE[i,i] = Hevals[i+1] - Hevals[1]
    end
    return ΔE
end

function cal_Kernel(J,h,MF,k)
    Hevals, Hevecs = Partlal_Basis_hex(J,h,MF)
    Aevals, Aevecs = Partial_basis_ABCsite(J,h,MF,"A")
    Bevals, Bevecs = Partial_basis_ABCsite(J,h,MF,"B")
    Cevals, Cevecs = Partial_basis_ABCsite(J,h,MF,"C")

    #那須先生のテキストのJ_{AB}をJAB, J_{A\Bar{B}}をJABBとする。
    J6A_k=Kernel_J("A",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6AB_k=Kernel_J("A",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6A_mk_T = transpose(Kernel_J("A",0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) #mk は-kのこと。
    J_6AB_mk_C = conj(Kernel_J("A",1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6B_k=Kernel_J("B",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6BB_k=Kernel_J("B",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6B_mk_T = transpose(Kernel_J("B" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) 
    J_6BB_mk_C = conj(Kernel_J("B" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6C_k=Kernel_J("C" ,0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6CB_k=Kernel_J("C" ,1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6C_mk_T = transpose(Kernel_J("C" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J_6CB_mk_C = conj(Kernel_J("C" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))

    ΔE_Hex= calΔE_hex(Hevals)
    ΔEA = Aevals[2] - Aevals[1]
    ΔEB = Bevals[2] - Bevals[1]
    ΔEC = Cevals[2] - Cevals[1]

    M_k = zeros(ComplexF64,132,132)
    M_k =  [zeros(63,63) J_6AB_k     J_6BB_k     J_6CB_k     zeros(63,63) J6A_k      J6B_k      J6C_k;
            zeros(1,63)  0           0           0           J6A_mk_T     0          0          0    ;
            zeros(1,63)  0           0           0           J6B_mk_T     0          0          0    ;
            zeros(1,63)  0           0           0           J6C_mk_T     0          0          0    ;
            zeros(63,63) zeros(63,1) zeros(63,1) zeros(63,1) zeros(63,63) J_6AB_mk_C J_6BB_mk_C J_6CB_mk_C; 
            zeros(1,132);
            zeros(1,132);
            zeros(1,132)]
    M_k = M_k + M_k'
    
    inner=  [ΔE_Hex zeros(63, 3);
            zeros(1,63) ΔEA zeros(1, 2);
            zeros(1,64) ΔEB zeros(1,1);
            zeros(1,65) ΔEC]
    Diag = kron(I, inner)
    return M_k = M_k + Diag    
end

#バンドプロット部分
#固有値の小さい順に返す
function disp(J,h,MF,k)
    w,_ = BogoliubovModule.Bogoliubov(cal_Kernel(J,h,MF,k))
    w_physical = real.(w)[1:66]
    return sort(w_physical)
end


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
        val = disp(J, h, MF, k)
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
savefig(p, "testdispersion_m=9over9.png")

end



    






