function idx_to_coord(idx, sz, edge_frac=1/7)
    # round(Int, (idx==1) ?  sz.*edge_frac .+1 : sz.*(1 .- edge_frac) .+1)
end

"""
    get_default_markers(image)
    returns the default marker positions of a given `image` dataset.
    The default is at 25% and 75% along each dimension.
"""
function get_default_markers(image, grid_size=(5,5))
    # NIdx = Tuple(2 .*ones(Int, ndims(image)))
    
    # [ 1 .+ round.(Int, size(image) ./ grid_size .* Tuple(idx)) for idx in CartesianIndices(grid_size)]
    # [idx_to_coord.(Tuple(idx), size(image)) [:]]
    # offset = (1,1) # 1 .+ size(image) .÷ (2 .*grid_size)
    # stepsize = (size(image) .- offset .+ 1) ./ grid_size
    # nodes = collect(offset[n]:stepsize[n]:size(image,n).-offset[n] for n=1:ndims(image))
    nodes = map(axes(image), grid_size, size(image)) do ax, g, s
        range(first(ax), stop=last(ax), length=g)
        # range(first(ax), stop=last(ax), length=g)
    end
    return nodes
end

"""
    extract_patches(img, markers=nothing, patch_size=max.(size(img) .÷5,5))
    extracts pathes, usually near at the corners as defined by the patch center positions
    `markers`. The patch size can be specified by `patch_size`.
#Arguments
+ img:          image from which the patches are extracted using `select_region` from the NDTools package.
+ markers:      a vector of center positions where the alignment should be determined. If `nothing` is provided, positions at 25% and 75% of each coordinates are chosen.
+ patch_size:   tuple denoting the size of each patch

"""
function extract_patches(img, markers=nothing; grid_size=(5,5), patch_size=max.(size(img) .÷ 5,5), avoid_border=true)
    patches = []
    markers = let
        if isnothing(markers)
            nodes = get_default_markers(img, grid_size)
            # omit the borders markers
            if avoid_border
                mar = [floor.(Int,(nodes[1][n], nodes[2][m])) for m = 2:grid_size[2]-1 for n = 2:grid_size[1]-1]
            else
                mar = []
                for m = 1:grid_size[2] 
                    for n = 1:grid_size[1]
                        m1 = (nodes[1][n], nodes[2][m])
                        m2 = (nodes[1][clamp(n,1,grid_size[1]-1)], nodes[2][clamp(m,1,grid_size[2]-1)])
                        push!(mar,floor.(Int,(m1 .+ m2)./2))
                    end
                end
                # mar = [floor.(Int,(nodes[1][n], nodes[2][m])) for m = 1:grid_size[2] for n = 1:grid_size[1]]
                mar
            end

            # the line above leads to deformantions. This can be fixed.
        else
            markers
        end
    end
    for m in markers
        patch = select_region(img, new_size=patch_size, center=Tuple(m))
        push!(patches, patch)
    end
    return patches, markers # cat(patches..., dims=ndims(img)+1)
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

function get_shift(myshift)
    SMatrix{3,3}([1.0 0.0 myshift[1];0.0 1.0 myshift[2];0.0 0.0 1.0])
end

"""
    get_rotation(sz::Tuple, Φ)
    returns a rotation operation in homography coordinates by the angle `Φ` in rad, which accounts for the middle pixel kept unchanged.
"""
function get_rotation(sz::Tuple, Φ, zoom=1.0)
    midpos = (sz .÷2) .+1
    shift_mat = get_shift(.-midpos)
    rot_mat = SMatrix{3,3}([zoom .* cos(Φ) -zoom .* sin(Φ) 0; zoom .* sin(Φ) zoom .* cos(Φ) 0; 0.0 0.0 1.0])
    shift_mat2 = get_shift(midpos) 
    return shift_mat2 * rot_mat * shift_mat #  
end

function get_rotation(moving::AbstractMatrix, Φ)
    get_rotation(size(moving), Φ)
end

"""
    get_rigid_warp(pre_transform_params, asize)
    transfers the parameters as determined by the `fourier_mellin()` routine to a warp.
"""
function get_rigid_warp(params, asize)
    (α, zoom, myshift) = params
    M = get_rotation(asize, α, zoom) # [cos(α) -sin(α); sin(α) cos(α)]
    # the order of the line below may look surprising but the warps seem to work backwards
    M = M * SMatrix{3,3}([1.0 0.0 myshift[1];0.0 1.0 myshift[2];0.0 0.0 1.0])
    ϕ(x) = (M*[x...,1])[1:2] # AffineMap(M, myshift)
    return ϕ
end 

function band_pass(img, s, e) 
    gaussf(img,s) .- gaussf(img,e)
end

"""
    align_images(img_list; average_pos=true, extra_shift=(0.25,0.25), band_pass_freq=0.25, grid_size=(10,10), patch_size=max.(size(img_list[1]) .÷ grid_size,5), tolerance=max.(size(img_list[1]) .÷ grid_size,5), avoid_border=false, warn_norm=10.0)

aligns a list of image (`img_list`) to the first image (the reference) using deformations. To this aim, the image is band-pass filtered, aligned rigidly via a Fourier-Mellin transformation
and then a patch-based alignment is performed. It is important that the images have some information almost everywhere.
Returned ia a tuple of the aligned images and a vector of warps to perform the transform for any other images via the "warp" call.


#Arguments
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

    warps = find_deformations(rigidly_aligned[1], rigidly_aligned[2:4], grid_size, average_pos=average_pos, extra_shift =extra_shift, patch_size=patch_size, pre_transform_params=params[2:end], avoid_border=avoid_border, tolerance=tolerance, warn_norm=warn_norm)
    all_aligned = [replace_nan(warp(img_list[n], warps[n])) for n=1:length(img_list)]
    return all_aligned, warps 
end
