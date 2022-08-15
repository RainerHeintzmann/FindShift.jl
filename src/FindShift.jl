"""
    FindShift

contains tools to quickly find the shift between images (pixel and sub-pixel precision) and to register images via rigid,
homography and non-rigid (patch-based) transforms.
"""
module FindShift
using FourierTools, IndexFunArrays, NDTools, Optim, Zygote, LinearAlgebra, ChainRulesCore, Statistics
using FFTW
using LazyArrays, StaticArrays
using RegisterDeformation, CoordinateTransformations
using PSFDistiller # currently needed for gaussf

export abs2_ft_peak, sum_exp_shift, find_ft_peak, correlate, beautify, get_subpixel_peak, align_stack, optim_correl
export find_shift, find_shift_iter, shift_cut, separable_view, arg_n
export determine_homography_warps, locate_patches, extract_patches, get_default_markers
export fourier_mellin, fourier_mellin_align
export find_deformations, align_images
export extract_sub_images, replace_nan

include("utils.jl")
include("exp_shifts.jl")
include("transforms.jl")
include("fourier_mellin.jl")
include("find_shift.jl")
include("find_deformations.jl")
include("homography_warp.jl")
include("extract_sub_images.jl")

end # module
