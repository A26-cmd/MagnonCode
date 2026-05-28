module iterationModule

using SparseArrays
using OffsetArrays
using LinearAlgebra

N=9
ndim=2^N
I = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
X = sparse([0.0 1.0+0.0im; 1.0+0.0im 0.0])
Y = sparse([0.0 -1.0im; 1.0im 0.0])
Z = sparse([1.0+0.0im 0.0; 0.0 -1.0+0.0im])

function times(list)
    list = OffsetArray(list, 0:N-1)
    list[0] = kron(list[0], list[1])
    for i in 2:length(list)-1
        list[0] = kron(list[0], list[i])
    end
    return list[0]
end

function set_spins(sites, σs,N=9)
    list_mats = fill(I, N)
    list_mats = OffsetArray(list_mats, 0:N-1)
    for (site, σ) in zip(sites, σs)
        list_mats[site] = σ
    end
    return list_mats
end

function I_ndim(N=9)
    list = fill(I,N)
    return times(list)
end
function S_idotS_j_op(i,j,J)
    return J*times(set_spins([i,j],[X,X]))+J*times(set_spins([i,j],[Y,Y]))+J*times(set_spins([i,j],[Z,Z]))

end

function x_op(i,val)
    return val*times(set_spins([i],[X]))
end

function y_op(i,val)
    return val*times(set_spins([i],[Y]))
end

function z_op(i,val)
    return val*times(set_spins([i],[Z]))
end
function MFhamiltonian(J,h,MF)
    MF = OffsetArray(MF, 0:17)
    # MFdataの順番：<S1245>_x,<S_1245>_y,<S1245>_z,<S1346>_x,<S1346>_y,<S1346>_z,<S2356>_x,<S2356>_y,<S2356>_z,<SA>_x,<SA>_y, <SA>_z,<SB>_x,<SB>_y,<SB>_z,<SC>_x,<SC>_y,<SC>_z
    # 0~5サイトは研究ノートの通り、6,7,8がA,B,C
    H = zeros(ComplexF64, ndim, ndim)
    #六角形部分
    for i in 0:4
        H += S_idotS_j_op(i,i+1,J)
    end
    H+= S_idotS_j_op(0,5,J)
    H += x_op(6,MF[0]*J)+
    y_op(6,MF[1]*J)+
    z_op(6,MF[2]*J-h)+
    x_op(7,MF[3]*J)+
    y_op(7,MF[4]*J)+
    z_op(7,MF[5]*J-h)+
    x_op(8,MF[6]*J)+
    y_op(8,MF[7]*J)+
    z_op(8,MF[8]*J-h)+
    
    x_op(0,(MF[9]+MF[12])*J)+
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

   H += -J*(MF[0]*MF[9]+MF[1]*MF[10]+MF[2]*MF[11]+MF[3]*MF[12]+MF[4]*MF[13]+MF[5]*MF[14]+MF[6]*MF[15]+MF[7]*MF[16]+MF[8]*MF[17])*I_ndim()
    return H
end



function solve_mf_hamiltonian(J,h,MF)
    H = MFhamiltonian(J,h,MF)
    return evals, evecs = eigen(H)
end

function MFcalT(evals, evecs,T,operator)

    if T==0
        gs = evecs[:,1]
        return real(dot(gs, operator * gs))
    else
        Z=0
        for i in 1:length(evals)
            Z += exp(-evals[i]/T)
        end
        expectation_value = 0
        for i in 1:length(evals)
            expectation_value += exp(-evals[i]/T)*real(dot(evecs[:,i], operator * evecs[:,i]))
        end
        return expectation_value/Z
    end
end


function returnMF(J,h,MF,T)
    evals, evecs = solve_mf_hamiltonian(J,h,MF)
    tmp = OffsetArray(zeros(18), 0:17)
    # 平均場の更新
    MFout = OffsetArray(zeros(18), 0:17)
    tmp[0] =real(MFcalT(evals,evecs,T, x_op(0, 1.0)))
    tmp[1] = real(MFcalT(evals,evecs,T, y_op(0, 1.0)))
    tmp[2] = real(MFcalT(evals,evecs,T, z_op(0, 1.0)))
    tmp[3] = real(MFcalT(evals,evecs,T, x_op(1, 1.0)))
    tmp[4] = real(MFcalT(evals,evecs,T, y_op(1, 1.0)))
    tmp[5] = real(MFcalT(evals,evecs,T, z_op(1, 1.0)))
    tmp[6] = real(MFcalT(evals,evecs,T, x_op(2, 1.0)))
    tmp[7] = real(MFcalT(evals,evecs,T, y_op(2, 1.0)))
    tmp[8] = real(MFcalT(evals,evecs,T, z_op(2, 1.0)))
    tmp[9] = real(MFcalT(evals,evecs,T, x_op(3, 1.0)))
    tmp[10] = real(MFcalT(evals,evecs,T, y_op(3, 1.0)))
    tmp[11] = real(MFcalT(evals,evecs,T, z_op(3, 1.0)))
    tmp[12] = real(MFcalT(evals,evecs,T, x_op(4, 1.0)))
    tmp[13] = real(MFcalT(evals,evecs,T, y_op(4, 1.0)))
    tmp[14] = real(MFcalT(evals,evecs,T, z_op(4, 1.0)))
    tmp[15] = real(MFcalT(evals,evecs,T, x_op(5, 1.0)))
    tmp[16] = real(MFcalT(evals,evecs,T, y_op(5, 1.0)))
    tmp[17] = real(MFcalT(evals,evecs,T, z_op(5, 1.0)))

    MFout[0]=tmp[0]+tmp[3]+tmp[9]+tmp[12];
    MFout[1]=tmp[1]+tmp[4]+tmp[10]+tmp[13];
    MFout[2]=tmp[2]+tmp[5]+tmp[11]+tmp[14];
    MFout[3]=tmp[0]+tmp[6]+tmp[9]+tmp[15];
    MFout[4]=tmp[1]+tmp[7]+tmp[10]+tmp[16];
    MFout[5]=tmp[2]+tmp[8]+tmp[11]+tmp[17];
    MFout[6]=tmp[3]+tmp[6]+tmp[12]+tmp[15];
    MFout[7]=tmp[4]+tmp[7]+tmp[13]+tmp[16];
    MFout[8]=tmp[5]+tmp[8]+tmp[14]+tmp[17];

    MFout[9]= real(MFcalT(evals,evecs,T, x_op(6, 1.0)))
    MFout[10]= real(MFcalT(evals,evecs,T, y_op(6, 1.0)))
    MFout[11]= real(MFcalT(evals,evecs,T, z_op(6, 1.0)))
    MFout[12]= real(MFcalT(evals,evecs,T, x_op(7, 1.0)))
    MFout[13]= real(MFcalT(evals,evecs,T, y_op(7, 1.0)))
    MFout[14]= real(MFcalT(evals,evecs,T, z_op(7, 1.0)))
    MFout[15]= real(MFcalT(evals,evecs,T, x_op(8, 1.0)))
    MFout[16]= real(MFcalT(evals,evecs,T, y_op(8, 1.0)))
    MFout[17]= real(MFcalT(evals,evecs,T, z_op(8, 1.0)))
    return evals, evecs, MFout,tmp

end

function iteration(J, h, m_init,T; N_iter=1000000, eps=1e-8, α_val=0.4)
    m_init = OffsetArray(m_init, 0:17)
    
    m_current = copy(m_init)

    evals = nothing
    evecs = nothing

    for i in 1:N_iter
        evals, evecs, m_new,tmp = returnMF(J, h, m_current, T)
        if i % 100 == 0
            println("Iteration $i: Mean fields = ", m_new)
        end
        if all(abs.(m_new .- m_current) .< eps)
            println("--- All components converged! ---")
            return evals, evecs, m_new,tmp
        end

        # 3. 線形混合 (Linear Mixing)
        # 次のステップの入力を、古い値(m_current)と新しい値(m_new)のブレンドにする
        # m_current = α * 古い値 + (1 - α) * 新しい値
        m_current = α_val .* m_current .+ (1.0 - α_val) .* m_new
    end

    println("Warning: Did not converge within $N_iter iterations.")
    return evals, evecs, m_current
end

function assert_MFapprox(J,h,evecs,evals,MF)

    H=zeros(ComplexF64, ndim, ndim)
    for i in 0:4
        H += S_idotS_j_op(i,i+1,J)
    end
    H+= S_idotS_j_op(0,5,J)
    gs=evecs[:,1]
    energy=real(dot(gs, H * gs))
    Stotz = (MF[2]+MF[5]+MF[8])/2+MF[11]+MF[14]+MF[17]

    diff = evals[1]-(energy-h*Stotz+J*(MF[0]*MF[9]+MF[1]*MF[10]+MF[2]*MF[11]+MF[3]*MF[12]+MF[4]*MF[13]+MF[5]*MF[14]+MF[6]*MF[15]+MF[7]*MF[16]+MF[8]*MF[17]))
end

T=0
J=1.0
h=1.0
m_current = zeros(Float64, 18)


m_current[2]=4.0

m_current[5]=4.0


m_current[8]=4.0

m_current[11]=1.0

m_current[14]=1.0

m_current[17]=1.0
#evals, evecs, m_final,tmp = iteration(J/4, h, m_current, T)#正しくスピンの値を表現するにはJを4でわる必要がある
#println("基底状態のエネルギー: ", evals[1])
#println("Final mean fields: ", m_final)
#println("六角形上のスピンvalues: ", tmp)
#println("平均場近似の妥当性: ", assert_MFapprox(J/4,h,evecs,evals,m_final))
#println("evecs:",evecs)

end