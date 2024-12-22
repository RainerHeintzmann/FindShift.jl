"""
    FindShift

contains tools to quickly find the shift between images (pixel and sub-pixel precision) and to register images via rigid,
homography and non-rigid (patch-based) transforms.
"""
module FindShift
using FourierTools, IndexFunArrays, NDTools, LinearAlgebra, ChainRulesCore, Statistics
using Optim, Zygote # for iterative optimization of positions
using FFTW
using StaticArrays # for fast arrays
# using RegisterDeformation  # only used for the warp function, which should be replaced by Interpolations.jl
using ThinPlateSplines # for the TPS deformation
using StaticArrays # for fast arrays
# using CoordinateTransformations
# using PSFDistiller # currently needed for gaussf
using SeparableFunctions
using ImageMorphology # for distance-transform based weighting and erosion of masks
using View5D # for the interactive subpixel peak finder we need get_positions

export abs2_ft_peak, sum_exp_shift, find_ft_peak, correlate, beautify, get_subpixel_peak, align_stack, optim_correl
export get_rel_subpixel_correl
export find_shift, find_shift_iter, shift_cut, separable_view, arg_n
export determine_homography_warps, locate_patches, extract_patches, get_default_markers
export fourier_mellin, fourier_mellin_align
export find_deformations, align_images
export extract_sub_images, replace_nan
export inpaint # see also the code in FourierTools for the version without mask
export find_pos, stitch, apply_warp, get_subpixel_correl

# export find_rel_pos, inpaint_imgs, minimize_distances, limit_strain!, get_strain, get_lin_idx, direction_tuple, set_border!, is_mask_array, get_dt_weights

include("utils.jl")
include("exp_shifts.jl")
include("transforms.jl")
include("fourier_mellin.jl")
include("find_shift.jl")
include("find_deformations.jl")
include("homography_warp.jl")
include("extract_sub_images.jl")
include("stitching.jl")

end # module
