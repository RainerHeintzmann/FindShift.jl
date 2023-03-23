"""
    get_default_markers(image)
    returns the default marker positions of a given `image` dataset.
    The default is at 25% and 75% along each dimension.
"""
function get_default_markers(image, grid_size=(5,5))
    nodes = map(axes(image), grid_size, size(image)) do ax, g, s
        range(first(ax), stop=last(ax), length=g)
    end
    return nodes
end


"""
    find_deformations(fixed, movings, grid_size=(5,5); patch_size=max.(size(fixed) .÷ grid_size,5), tolerance = max.(size(fixed) .÷ grid_size,5), pre_transform_params=nothing)
    determines the warps between a refrence image `fixed` and one or multiple images `moving` to align to that reference image.

"""
function find_deformations(fixed, movings, grid_size=(11, 11); average_pos=false, extra_shift=(0.25,0.25), patch_size=max.(size(fixed) .÷ grid_size,5), tolerance = max.(size(fixed) .÷ grid_size,5), pre_transform_params=nothing, avoid_border=false, warn_norm=10.0)
    nodes = get_default_markers(fixed, grid_size)
    ref_patches, markers = extract_patches(fixed, patch_size=patch_size; grid_size=grid_size, avoid_border=avoid_border)

    warps = []
    ni = 1
    all_shifts = [] # stores all the shifts
    sum_shifts = zeros((2,grid_size...)) 
    for moving in movings
        search_in, markers = extract_patches(moving, patch_size=patch_size .+ tolerance; grid_size=grid_size, avoid_border=avoid_border)
        myshifts = zeros((2,grid_size...))
        # does not account for the pre-alignment
        w = let
            if isnothing(pre_transform_params)
                nothing
            else
                FindShift.get_rigid_warp(pre_transform_params[ni], size(moving))
            end
        end
        n = 1
        # iterate through the patches
        for ci in CartesianIndices(grid_size)
            ci = Tuple(ci)
            # measures the difference compared to the middle of each patch, since it expands to the larger size.
            if avoid_border
                cn = clamp.(ci, 2, grid_size.-1)
            else
                cn = ci
            end
            if ci == cn # only calculate shift where needed and copy the border shifts
                pr = ref_patches[n]
                ps = search_in[n]
                myshift = find_shift_iter(pr, ps)
                if norm(myshift) > warn_norm
                    @warn "large shift $(norm(myshift))>$(warn_norm) at postion $(ci) of $(grid_size)."
                end
                sum_shifts[:,ci...] .+= .-myshift
                if isnothing(pre_transform_params)
                    myshifts[:,ci...] .= .- myshift
                else # account for the rigid pre-transform 
                    marker = markers[n]
                    #@show myshift
                    #@show (w(marker) .- marker) .- myshift
                    myshifts[:,ci...] .= (w(marker) .- marker) .- myshift
                end
                n = n + 1
            end
        end
        # copy inner bit to borders 
        if avoid_border
            for ci in CartesianIndices(grid_size)
                ci = Tuple(ci)
                cn = clamp.(ci, 2, grid_size.-1)
                myshifts[:,ci...] .= myshifts[:,cn...]
            end
        end
        # @show myshifts
        push!(all_shifts, myshifts)
        ni += 1
    end
    mean_shifts = sum_shifts ./ (1 + length(movings)) # the 1 accounts for the zero shift of the reference image
    if average_pos
        warps = [GridDeformation(myshifts .+ mean_shifts .+ extra_shift, nodes) for myshifts in all_shifts]
        warps = [GridDeformation(mean_shifts .+ extra_shift, nodes), warps...]
    else
        warps = [GridDeformation(myshifts, nodes) for myshifts in all_shifts]
        if norm(extra_shift) == 0
            warps = [IdentityTransformation, warps...]
        else
            warps = [GridDeformation(mean_shifts.*0 .+ extra_shift, nodes), warps...]
        end
    end
    return warps
end



"""
    locate_patches(patches, large_image)
    locates each patch in a vector of image `patches` in a `large_image`.
    returned is vector of the sup-pixel center positions of the located patches and 
    a large image with the patches shifted to the found positions. 
    Note that this algorithm works with sub-pixel precision.
"""
function locate_patches(patches, large_image)
    mymid = size(large_image).÷2 .+1
    located_pos = []
    shifted = zeros(size(large_image))
    for snippet in patches
        myshift = find_shift_iter(select_region(large_image, new_size=size(large_image).+size(snippet)), snippet)
        push!(located_pos, myshift .+ mymid)
        shifted .+= shift(select_region(snippet, new_size=size(large_image)), myshift)
    end
    return located_pos, shifted 
end

"""
    align_images(img_list; average_pos=true, extra_shift=(0.25,0.25), band_pass_freq=0.25, grid_size=(10,10), patch_size=max.(size(img_list[1]) .÷ grid_size,5), tolerance=max.(size(img_list[1]) .÷ grid_size,5), avoid_border=false, warn_norm=10.0)

aligns a list of image (`img_list`) to the first image (the reference) using deformations. To this aim, the image is band-pass filtered, aligned rigidly via a Fourier-Mellin transformation
and then a patch-based alignment is performed. It is important that the images have some information almost everywhere.
Returned ia a tuple of the aligned images and a vector of warps to perform the transform for any other images via the "warp" call.


#Arguments
+ `img_list`: a vector of (2D-) images to align. the first is the reference image.
+ `average_pos`:   if true, all images (including the first one) will be aligned to the average deformation (not accounting for the rigid pre-alignment)
+ `extra_shift``:   an extra shift to apply to all images (including the first one). The default (0.25,0.25) aims to generate a little bit of interpolation, to make all aligned images more comparable in quality.
+ `band_pass_freq`:  the relative center frequency of the band pass (compared to the Nyquist/border frequency)
+ `grid_size`:      the number of evenly spaced grid points for which the alignment parameters are determined.
+ `patch_size`:     the size of each patch in the (first) reference image to compare
+ `tolerance`:     the extra size that the patches of the images to align get as search space. Usually this can be kept small.
+ `avoid_border`:   if `true` the no alignment for the border patches will be determined but the neighbour parameters (after rigid alignment) will be used instead
+ `warn_norm`:     a shift distance (in pixels) after pre-alignment above which a warning will be issued. 
"""
function align_images(img_list; average_pos=true, extra_shift=(0.25, 0.25), band_pass_freq=0.25, grid_size=(10,10), patch_size=max.(size(img_list[1]) .÷ grid_size,5), tolerance=max.(size(img_list[1]) .÷ grid_size,5), avoid_border=false, warn_norm=10.0)
    filtered = []
    s = 0.8 ./ band_pass_freq
    e = 1.2 ./ band_pass_freq

    for img in img_list
        f = band_pass(img, s, e)
        push!(filtered, f)
    end
    reference = filtered[1]
    movings = filtered[2:end]
    rigidly_aligned, params = fourier_mellin_align(reference, movings)
    # return rigidly_aligned, params

    warps = find_deformations(rigidly_aligned[1], rigidly_aligned[2:4], grid_size, average_pos=average_pos, extra_shift =extra_shift, patch_size=patch_size, pre_transform_params=params[2:end], avoid_border=avoid_border, tolerance=tolerance, warn_norm=warn_norm)
    all_aligned = [replace_nan(warp(img_list[n], warps[n])) for n=1:length(img_list)]
    return all_aligned, warps 
end
