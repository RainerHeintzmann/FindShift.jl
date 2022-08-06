module FindShift
export abs2_ft_peak, sum_exp_shift, find_ft_peak, correlate, beautify, get_subpixel_peak, align_stack, optim_correl
export find_shift_iter, shift_cut, separable_view, arg_n
export determine_homography_warps, locate_patches, extract_patches, get_default_markers

using FourierTools, IndexFunArrays, NDTools, Optim, Zygote, LinearAlgebra, ChainRulesCore, Statistics
using FFTW
using LazyArrays, StaticArrays

include("find_shift.jl")
include("homography_warp.jl")

end # module
