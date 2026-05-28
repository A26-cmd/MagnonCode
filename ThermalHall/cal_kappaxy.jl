#koyama Nasu論文に基づいて熱ホールを計算

module kappaxycalModule

const N_mat = 132
const M = div(N_mat,2)
const J = 1.0

#六角形中心とA ,B,Cサイトの相対位置 kagomeの最小単位胞a=1
const δA=[-3/4,sqrt(3)/4]
const δB=[3/4, sqrt(3)/4]
const δC =[0,sqrt(3)/2]
const δAx = -3/4
const δAy = sqrt(3)/4
const δBx = 3/4
const δBy= sqrt(3)/4
const δCx=0
const δCy=sqrt(3)/2

const η=0.001 #縮退している時の発散を回避




include("Bogoliubov.jl")
include("Bose.jl")
include("Brillouin_sum.jl")
include("c2(x)_nasuver.jl")
include("Magnon.jl")
using .BogoliubovModule
using .BoseModule
using .BrillouinsumModule
using .C2hokanModule
using .Magnonbandmodule
using StaticArrays
using LinearAlgebra
using Roots
using DataFrames
using CSV
using Polylogarithms
using SparseArrays
using OffsetArrays
using Base.Threads

I = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
X = sparse([0.0 1.0+0.0im; 1.0+0.0im 0.0])
Y = sparse([0.0 -1.0im; 1.0im 0.0])
Z = sparse([1.0+0.0im 0.0; 0.0 -1.0+0.0im])


function Bibun_Kernel_x(J,h,MF,k, Hevecs,Aevecs,Bevecs, Cevecs)

    #那須先生のテキストのJ_{AB}をJAB, J_{A\Bar{B}}をJABBとする。正確にはこれらを微分したもの。
    J6A_k=im*δAx*Kernel_J("A",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6AB_k=im*δAx*Kernel_J("A",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6A_mk_T = transpose(-im*δAx*Kernel_J("A",0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) #mk は-kのこと。
    J_6AB_mk_C = conj(-im*δAx*Kernel_J("A",1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6B_k=im*δBx*Kernel_J("B",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6BB_k=im*δBx*Kernel_J("B",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6B_mk_T = transpose(-im*δBx*Kernel_J("B" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) 
    J_6BB_mk_C = conj(-im*δBx*Kernel_J("B" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6C_k=im*δCx*Kernel_J("C" ,0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6CB_k=im*δCx*Kernel_J("C" ,1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6C_mk_T = transpose(-im*δCx*Kernel_J("C" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J_6CB_mk_C = conj(-im*δCx*Kernel_J("C" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))

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
    return M_k  
end

function Bibun_Kernel_y(J,h,MF,k, Hevecs,Aevecs,Bevecs, Cevecs)

    #那須先生のテキストのJ_{AB}をJAB, J_{A\Bar{B}}をJABBとする。正確にはこれらを微分したもの。
    J6A_k=im*δAy*Kernel_J("A",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6AB_k=im*δAy*Kernel_J("A",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6A_mk_T = transpose(-im*δAy*Kernel_J("A",0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) #mk は-kのこと。
    J_6AB_mk_C = conj(-im*δAy*Kernel_J("A",1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6B_k=im*δBy*Kernel_J("B",0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6BB_k=im*δBy*Kernel_J("B",1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6B_mk_T = transpose(-im*δBy*Kernel_J("B" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs)) 
    J_6BB_mk_C = conj(-im*δBy*Kernel_J("B" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J6C_k=im*δCy*Kernel_J("C" ,0,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J_6CB_k=im*δCy*Kernel_J("C" ,1,k,J,Hevecs,Aevecs,Bevecs,Cevecs)
    J6C_mk_T = transpose(-im*δCy*Kernel_J("C" ,0,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))
    J_6CB_mk_C = conj(-im*δCy*Kernel_J("C" ,1,-k,J,Hevecs,Aevecs,Bevecs,Cevecs))

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
    return M_k  
end


function cal_Kernel(J,h,MF,k,Hevals, Hevecs, Aevals, Aevecs, Bevals,Bevecs,Cevals,Cevecs)

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

σ3= vcat(ones(M), -ones(M))

#小山さんの論文のΩnk
function omega_nk(J,h,n,kx,ky,MF,Hevals,Hevecs,Aevals,Aevecs,Bevals,Bevecs,Cevals,Cevecs,w,U)
    k = [kx, ky]
    sum = 0.0
    UX = U'*Bibun_Kernel_x(J,h,MF,k, Hevecs,Aevecs,Bevecs, Cevecs)*U
    UY = U'*Bibun_Kernel_y(J,h,MF,k, Hevecs,Aevecs,Bevecs, Cevecs)*U

    #数値安定化ver
    UX = (UX' + UX)/2
    UY = (UY' + UY)/2
    for m in 1:N_mat
        if m == n
            continue
        #=elseif real(w[m]-w[n]) < 1e-6 #バンド縮退が起こりそうな場合に和に含めない
            band_degeneracy =1
            #println("発散点あり kx=", kx, "ky=", ky, "n=", n, "m=", m) #縮退がある場合、めっちゃ遅くなる
            continue
        =#
        else
            #sum += imag((σ3[n]*σ3[m]*UX[n, m]*UY[m, n]) / (σ3[m]*w[m]-σ3[n]*w[n])^2)
            sum += imag((σ3[n]*σ3[m]*UX[n, m]*UY[m, n]) / ((w[m]-w[n])^2 + η^2))
        end
    end

    #println("sum:", sum, "w:",w) #動作確認
    return -2*sum, w
end



function kappa_xy(J,h,T,MF)
    println("h:",h, " T:", T, " threads:", nthreads())
    Hevals, Hevecs = Partlal_Basis_hex(J,h,MF)
    Aevals, Aevecs = Partial_basis_ABCsite(J,h,MF,"A")
    Bevals, Bevecs = Partial_basis_ABCsite(J,h,MF,"B")
    Cevals, Cevecs = Partial_basis_ABCsite(J,h,MF,"C")
    c2omega = (kx,ky) -> begin
        k = [kx, ky]
        w,U = BogoliubovModule.Bogoliubov(cal_Kernel(J,h,MF,k,Hevals, Hevecs, Aevals, Aevecs, Bevals,Bevecs,Cevals,Cevecs))
        sum =0.0
        for n in 1:M
            ω, wlist = omega_nk(J,h,n,kx,ky,MF,Hevals,Hevecs,Aevals,Aevecs,Bevals,Bevecs,Cevals,Cevecs,w,U)
            x = real(BoseModule.Bose(wlist[n],T))
            sum += (C2hokanModule.c2(x)-pi^2/3)*ω #発散をpi^2/3が抑制 これどうなの？
            #println("c2(x):",c2(x), " ω:", ω)
        end
        return sum
    end
    return -T*BrillouinsumModule.Brillouin_sum(c2omega) #kb, h_var =1
end

#テスト　5/9プラトーに相当
T=0.1
MF= [5.621108407631938e-17,-2.0202362464925032e-17,1.3336098378585997,2.3063556656340933e-17,1.140903001990507e-16,1.3336098378585994,-5.663145415265288e-17,7.57560500288796e-17,1.3336098378586003,-5.008661810287673e-17,-1.1659733641815842e-17,0.9999999999999999,-4.516950517184773e-17,-1.8806504357493256e-16,0.9999999999999999,-5.5400706204353994e-17,-8.299732066779644e-17,0.9999999999999999
]
MF = OffsetArray(MF, 0:length(MF)-1)
h=4.81203007518797
kapa=kappa_xy(J,h,T,MF)
println("kappaxy=", kapa)

end
