module C2hokanModule

using DelimitedFiles
using DataInterpolations

export c2, reload_data

const DATA_FILE = "func_nasu.txt"

# 補間関数を1つだけ保持
const C2_SPLINE = Ref{CubicSpline}()

"""
    _load_data()

func_nasu.txt を読み込み、c2(x) 用の CubicSpline を構築する
"""
function _load_data()
    if !isfile(DATA_FILE)
        error("エラー: データファイル '$DATA_FILE' が見つかりません。")
    end

    data = readdlm(DATA_FILE)

    x = vec(data[:, 1])
    y = vec(data[:, 2])

    # NaN → 0
    replace!(y, NaN => 0.0)

    # x 昇順ソート（数学的に必須）
    p = sortperm(x)
    x_sorted = x[p]
    y_sorted = y[p]

    # 補間関数生成
    C2_SPLINE[] = CubicSpline(y_sorted, x_sorted)

    println("完了: func_nasu.txt から c2(x) をロードしました。")
end

# 初期ロード
_load_data()

"""
    reload_data()

データファイルを再読み込み
"""
function reload_data()
    _load_data()
end

"""
    c2(x)

c2(x) を補間で返す
"""
function c2(x)
    spline = C2_SPLINE[]
    spline === nothing && error("補間関数が初期化されていません。")

    x_real = real(x)

    try
        val = spline(x_real)
        return isnan(val) ? 0.0 : val
    catch
        # 範囲外アクセスなど
        return 0.0
    end
end

end
