#最初に様々な磁場をsweepさせて見つかったプラトー領域の平均場のもとで最も安定な解を探す
include("julia_make_hamiltonian_diaglize_returnMF.jl")
using Roots
using DataFrames
using CSV
using .iterationModule

const platexs = 5
#プラトー状態の参照平均場読み込み
filename = "Plataux_MFlist.csv"
df = CSV.read(filename, DataFrame)
mf_matrix=Matrix(df)
#println(mf_matrix[1, : ])

nh = 400
h_list = range(0.0, 12.0, length=nh)
T=0.1
J=1.0
Nh = length(h_list)

#最も安定な解をプラトー領域の解から探してその平均ばを返す関数
function MostStableMF(J,h,T)
    GSvaluedata = zeros(platexs)
    MFdata = zeros(Float64, platexs,18)
    for i in 1:platexs
        MF=mf_matrix[i, : ]
        evals, evecs, result, tmp = iterationModule.iteration(J,h,MF,T)
        GSvaluedata[i] = evals[1]
        for k in 1:18
            MFdata[i,k] = result[k-1]#resultはOffsetarray
        end
    end
    idx = argmin(GSvaluedata)

    return MFdata[idx, :]
end    
#書き込むファイルを準備
filename = "./MostStabilizeMFanswerT=$T.csv"

S1245x_list  = zeros(Float64, Nh)
S1245y_list  = zeros(Float64, Nh)
S1245z_list  = zeros(Float64, Nh)
S1346x_list  = zeros(Float64, Nh)
S1346y_list  = zeros(Float64, Nh)
S1346z_list  = zeros(Float64, Nh)
S2356x_list  = zeros(Float64, Nh)
S2356y_list  = zeros(Float64, Nh)
S2356z_list  = zeros(Float64, Nh)
SAx_list = zeros(Float64, Nh)
SAy_list = zeros(Float64, Nh)
SAz_list = zeros(Float64, Nh)
SBx_list = zeros(Float64, Nh)
SBy_list = zeros(Float64, Nh)
SBz_list = zeros(Float64, Nh)
SCx_list = zeros(Float64, Nh)
SCy_list = zeros(Float64, Nh)
SCz_list = zeros(Float64, Nh)

MF=zeros(Float64,18)

error_flag = false


CSV.write(
    filename,
    DataFrame(
        h = Float64[],
        S1245x = Float64[],
        S1245y = Float64[],
        S1245z = Float64[],
        S1346x = Float64[],
        S1346y = Float64[],
        S1346z = Float64[],
        S2356x = Float64[],
        S2356y = Float64[],
        S2356z = Float64[],
        SAx = Float64[],
        SAy = Float64[],
        SAz = Float64[],
        SBx = Float64[],
        SBy = Float64[],
        SBz = Float64[],
        SCx = Float64[],
        SCy = Float64[],
        SCz = Float64[]
    )
)

for i in 1:Nh
    h=h_list[i]
    if error_flag
        result = (
            1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
        )
    else
        try
            result = MostStableMF(J, h, T)#いまスピンじゃなくパウリ行列で実装しているからJ→J/4にする必要があるかも
            if result === nothing
                throw(ErrorException("iteration returned nothing"))
            end
        catch e
            println(e)
            global error_flag = true
            result = (
                1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
            )
        end
    end

    S1245x_list[i]  = result[1]
    S1245y_list[i]  = result[2]
    S1245z_list[i]  = result[3]
    S1346x_list[i]  = result[4]
    S1346y_list[i]  = result[5]
    S1346z_list[i]  = result[6]         
    S2356x_list[i]  = result[7]
    S2356y_list[i]  = result[8]
    S2356z_list[i]  = result[9]
    SAx_list[i]  = result[10]
    SAy_list[i]  = result[11]
    SAz_list[i]  = result[12]
    SBx_list[i]  = result[13]
    SBy_list[i]  = result[14]
    SBz_list[i]  = result[15]
    SCx_list[i]  = result[16]
    SCy_list[i]  = result[17]
    SCz_list[i]  = result[18]

    # ===== 逐次CSV追記 =====
    CSV.write(
        filename,
        DataFrame(
            h = [h_list[i]],
            S1245x = [S1245x_list[i]],
            S1245y = [S1245y_list[i]],
            S1245z = [S1245z_list[i]],
            S1346x = [S1346x_list[i]],
            S1346y = [S1346y_list[i]],
            S1346z = [S1346z_list[i]],
            S2356x = [S2356x_list[i]],
            S2356y = [S2356y_list[i]],
            S2356z = [S2356z_list[i]],
            SAx = [SAx_list[i]],
            SAy = [SAy_list[i]],
            SAz = [SAz_list[i]],
            SBx = [SBx_list[i]],
            SBy = [SBy_list[i]],
            SBz = [SBz_list[i]],
            SCx = [SCx_list[i]],
            SCy = [SCy_list[i]],
            SCz = [SCz_list[i]],
        ),
        append = true
    )

end




    

 