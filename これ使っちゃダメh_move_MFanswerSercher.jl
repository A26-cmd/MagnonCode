#前の平均場の値を次の値に入れてsweepするだけ
#様々なプラトーの初期状態を比較して最も安定な状態を探すものではない。

include("julia_make_hamiltonian_diaglize_returnMF.jl")
using Roots
using DataFrames
using CSV
using .iterationModule
nh = 400
h_list = range(0.0, 12.0, length=nh)
T=0.1
J=1.0
Nh = length(h_list)
#書き込むファイルを準備
filename = "./ManyMFanswerT=$T.csv"

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

#h=0の初期値
a = [6.518231427675172e-15, 8.44703272050834e-6, -1.3333333333065802, 7.20279597632376e-15, 8.447032720589756e-6, -1.3333333333065749, 6.47439704571354e-15, 8.447032721222966e-6, -1.3333333333065758, -5.235715514298052e-15, -6.295752791030852e-6, 0.999999999980182, -4.087616372166475e-15, -6.295752791519377e-6, 0.9999999999801826, -5.8069813301182046e-15, -6.295752791669194e-6, 0.9999999999801827]
S1245x_list[1]  = a[1]
S1245y_list[1]  = a[2]
S1245z_list[1]  = a[3]
S1346x_list[1]  = a[4]
S1346y_list[1]  = a[5]
S1346z_list[1]  = a[6]
S2356x_list[1]  = a[7]
S2356y_list[1] = a[8]
S2356z_list[1] = a[9]
SAx_list[1] = a[10]
SAy_list[1]= a[11]
SAz_list[1] = a[12]
SBx_list[1] = a[13]
SBy_list[1] = a[14]
SBz_list[1] = a[15]
SCx_list[1] = a[16]
SCy_list[1] = a[17]
SCz_list[1] = a[18]


for i in 2:Nh

    h = h_list[i]
    MF[1] = S1245x_list[i-1]
    MF[2] = S1245y_list[i-1]
    MF[3] = S1245z_list[i-1]
    MF[4] = S1346x_list[i-1]
    MF[5] = S1346y_list[i-1]
    MF[6] = S1346z_list[i-1]
    MF[7] = S2356x_list[i-1]
    MF[8] = S2356y_list[i-1]
    MF[9] = S2356z_list[i-1]
    MF[10] = SAx_list[i-1]
    MF[11] = SAy_list[i-1]
    MF[12] = SAz_list[i-1]
    MF[13] = SBx_list[i-1]
    MF[14] = SBy_list[i-1]
    MF[15] = SBz_list[i-1]
    MF[16] = SCx_list[i-1]
    MF[17] = SCy_list[i-1]
    MF[18] = SCz_list[i-1]

    if error_flag
        result = (
            1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
        )
    else
        try
            evals, evecs, result, tmp = iterationModule.iteration(J,h,MF,T)
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

    S1245x_list[i]  = result[0]
    S1245y_list[i]  = result[1]
    S1245z_list[i]  = result[2]
    S1346x_list[i]  = result[3]
    S1346y_list[i]  = result[4]
    S1346z_list[i]  = result[5]         
    S2356x_list[i]  = result[6]
    S2356y_list[i]  = result[7]
    S2356z_list[i]  = result[8]
    SAx_list[i]  = result[9]
    SAy_list[i]  = result[10]
    SAz_list[i]  = result[11]
    SBx_list[i]  = result[12]
    SBy_list[i]  = result[13]
    SBz_list[i]  = result[14]
    SCx_list[i]  = result[15]
    SCy_list[i]  = result[16]
    SCz_list[i]  = result[17]

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




    
