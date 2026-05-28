const SCRIPT_DIR = @__DIR__

function include_magnon_core()
    magnon_path = joinpath(SCRIPT_DIR, "Magnon.jl")
    lines = readlines(magnon_path)
    cutoff = findfirst(line -> occursin("# バンドの数を選べる", line), lines)
    cutoff === nothing && error("Magnon.jl の単発プロット部分の開始位置を見つけられません。")

    core_source = join(lines[1:cutoff-1], "\n") * "\nend\n"
    include_string(Main, core_source, magnon_path)
end

include_magnon_core()

using .Magnonbandmodule
using CSV
using DataFrames
using LaTeXStrings
using LinearAlgebra
using OffsetArrays
using Plots
using Printf

const J = 1.0
const N_bands = 6
const Ns = 50

const DEFAULT_CSV = joinpath(SCRIPT_DIR, "MostStabilizeMFanswerT=0.0.csv")
const DEFAULT_OUTPUT_DIR = joinpath(SCRIPT_DIR, "MagnonBands")

function band_path()
    Gamma = [0.0, 0.0]
    M_point = [0.0, sqrt(3)/2]
    K_point = [0.5, sqrt(3)/2]

    path1 = [(Gamma[1] + (M_point[1] - Gamma[1]) * (i / Ns),
              Gamma[2] + (M_point[2] - Gamma[2]) * (i / Ns)) for i in 0:Ns-1]
    path2 = [(M_point[1] + (K_point[1] - M_point[1]) * (i / Ns),
              M_point[2] + (K_point[2] - M_point[2]) * (i / Ns)) for i in 0:Ns-1]
    path3 = [(K_point[1] + (Gamma[1] - K_point[1]) * (i / Ns),
              K_point[2] + (Gamma[2] - K_point[2]) * (i / Ns)) for i in 0:Ns]

    L1 = sqrt(3)/2
    L2 = 0.5
    L3 = 1.0

    s1_plot = range(0.0, L1, Ns+1)[1:end-1]
    s2_plot = range(L1, L1 + L2, Ns+1)[1:end-1]
    s3_plot = range(L1 + L2, L1 + L2 + L3, Ns+1)[1:end]

    return [path1, path2, path3], vcat(s1_plot, s2_plot, s3_plot), L1, L2, L3
end

function make_plot(h, MF)
    paths, s_list, L1, L2, L3 = band_path()
    omega = [Float64[] for _ in 1:N_bands]

    for path in paths
        for (qx, qy) in path
            val = try
                Magnonbandmodule.disp(J, h, MF, [qx, qy])
            catch e
                if e isa PosDefException
                    fill(NaN, N_bands)
                else
                    rethrow()
                end
            end
            for b in 1:N_bands
                push!(omega[b], val[b])
            end
        end
    end

    all_values = vcat(omega...)
    finite_values = filter(isfinite, all_values)
    if isempty(finite_values)
        println("skipped h=$(h): all k points failed")
        return nothing
    end
    y_min = minimum(finite_values)
    y_max = maximum(finite_values)
    y_pad = (y_max - y_min) * 0.05

    ylim_bottom = y_min > -1e-5 ? 0.0 : y_min - y_pad
    ylim_top = y_max + y_pad

    p = plot(
        s_list, omega[1],
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

    for b in 2:N_bands
        plot!(p, s_list, omega[b], lw = 1.5, lc = :blue)
    end

    xticks_pos = [0.0, L1, L1 + L2, L1 + L2 + L3]
    xticks_line = [L1, L1 + L2]
    xticks_lbl = [L"\Gamma", L"\mathrm{M}", L"\mathrm{K}", L"\Gamma"]
    xticks!(p, xticks_pos, xticks_lbl)
    vline!(p, xticks_line, c = :black, ls = :dash, lw = 0.8, label = false)

    return p
end

function h_filename(h, row_index)
    index_text = lpad(row_index, 3, "0")
    h_text = @sprintf("%.12g", h)
    return "$(index_text)_magnon_band_h=$(h_text).png"
end

function plot_bands_from_csv(csv_path = DEFAULT_CSV, output_dir = DEFAULT_OUTPUT_DIR; max_rows = nothing)
    df = CSV.read(csv_path, DataFrame)
    if !("h" in names(df))
        error("CSVに h 列がありません: $(csv_path)")
    end

    mf_cols = names(df)[names(df) .!= "h"]
    if length(mf_cols) != 18
        error("MF列は18個必要です。見つかったMF列数: $(length(mf_cols))")
    end

    mkpath(output_dir)
    n_rows = isnothing(max_rows) ? nrow(df) : min(max_rows, nrow(df))

    for row_index in 1:n_rows
        row = df[row_index, :]
        h = Float64(row.h)
        MF = OffsetArray(Float64[row[col] for col in mf_cols], 0:length(mf_cols)-1)

        println("plotting row $(row_index)/$(n_rows), h=$(h)")
        p = make_plot(h, MF)
        if p !== nothing
            savefig(p, joinpath(output_dir, h_filename(h, row_index)))
        end
    end
end

csv_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CSV
output_dir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT_DIR
max_rows = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing

plot_bands_from_csv(csv_path, output_dir; max_rows = max_rows)
