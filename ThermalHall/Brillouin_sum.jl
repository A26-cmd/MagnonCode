module BrillouinsumModule #モジュールの作成

const N_k = 50 #もと５０
#const N_k = 32
#逆格子ベクトルを指定
const b1 = 2π*[2/3,0]
const b2 = 2π/3*[-1,sqrt(3)]
const para = range(-1/2,1/2,N_k+1)[2:end]
const Δn = para[2]-para[1]

using LinearAlgebra
using Base.Threads

# a=1とする。
function Brillouin_sum(f)
    partial_sums = zeros(Complex{Float64},nthreads())
    @threads for thread_id in 1:nthreads()
        local_sum = 0.0+0.0im
        for i in thread_id:nthreads():length(para)
            for j in 1:length(para)
                s = para[i]
                t = para[j]
                vec = (s-Δn/2)*b1+(t-Δn/2)*b2
                kx,ky = vec[1],vec[2]
                local_sum +=  f(kx,ky)
            end
        end
        partial_sums[thread_id] = local_sum
    end
    return sum(partial_sums)/N_k^2
end

function Brillouin_sum_vec(len::Integer,f)
    partial = [zeros(ComplexF64,len) for _ in 1:nthreads()]
    @threads for tid in 1:nthreads()
        acc = partial[tid]
        for i in tid:nthreads():length(para)
            for j in 1:length(para)
                s = para[i]
                t = para[j]
                vec = (s-Δn/2)*b1 + (t-Δn/2)*b2
                kx,ky = vec[1],vec[2]
                acc .+= f(kx,ky)
            end
        end
    end

    total = zeros(ComplexF64,len)
    for v in partial
        total .+= v
    end
    return total ./ N_k^2
end

end
