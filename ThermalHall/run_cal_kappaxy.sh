#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
julia --threads auto cal_kappaxy.jl
