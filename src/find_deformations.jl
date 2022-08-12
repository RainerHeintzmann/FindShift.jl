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
function extract_patches(img, markers=nothing; grid_size=(5,5), patch_size=max.(size(img) .÷ 5,5))
    patches = []
    markers = let
        if isnothing(markers)
            nodes = get_default_markers(img, grid_size)
            # omit the borders markers
            [floor.(Int,(nodes[1][n], nodes[2][m])) for m = 2:grid_size[2]-1 for n = 2:grid_size[1]-1]
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
function find_deformations(fixed, movings, grid_size=(11, 11); patch_size=max.(size(fixed) .÷ grid_size,5), tolerance = max.(size(fixed) .÷ grid_size,5), pre_transform_params=nothing)
    nodes = get_default_markers(fixed, grid_size)
    ref_patches, markers = extract_patches(fixed, patch_size=patch_size; grid_size=grid_size)
    @show size(markers)

    # all_aligned = [fixed,]
    warps = []
    ni = 1
    for moving in movings
        search_in, markers = extract_patches(moving, patch_size=patch_size .+ tolerance; grid_size=grid_size)
        myshifts = zeros((2,grid_size...))
        w = let
            if isnothing(pre_transform_params)
                nothing
            else
                get_rigid_warp(pre_transform_params[ni], size(moving))
            end
        end
        n = 1
        for ci in CartesianIndices(grid_size)
            ci = Tuple(ci)
            # measures the difference compared to the middle of each patch, since it expands to the larger size.
            cn = clamp.(ci, 2, grid_size.-1)
            if ci == cn # only calculate shift where needed and copy the border shifts
                pr = ref_patches[n]
                ps = search_in[n]
                myshift = find_shift_iter(pr, ps)
                myshifts[:,ci...] .= .-myshift
                if isnothing(pre_transform_params)
                    myshifts[:,ci...] .= .-myshift
                else # account for the rigid pre-transform 
                    marker = markers[n]
                    myshifts[:,ci...] .= (w(marker) .- marker) .- myshift
                end
                n = n + 1
            end
        end
        # copy inner bit to borders 
        for ci in CartesianIndices(grid_size)
            ci = Tuple(ci)
            cn = clamp.(ci, 2, grid_size.-1)
            myshifts[:,ci...] .= myshifts[:,cn...]
        end
        # @show myshifts
        ϕ = GridDeformation(myshifts, nodes)

        # warped = warp(moving, ϕ) # , axes(moving1)
        # push!(all_aligned, warped)

        # if !isnothing(pre_transform_params)            
        #     w = get_rigid_warp(pre_transform_params[ni], size(moving))
        #     push!(warps, ϕ) # (x) -> ϕi(w(x))
        # else
        #     push!(warps, ϕ)
        # end
        push!(warps, ϕ)
        ni += 1
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

function align_images(img_list; average_pos=true, band_pass_freq=0.25, grid_size=(10,10))
    filtered = []
    s = 0.8 ./ band_pass_freq
    e = 1.2 ./ band_pass_freq
    # do_resize(img) = select_region(img, new_size = round.(Int, size(img) .* grid_size ./ (grid_size .- 2)))
    do_resize(img) = img

    for img in img_list
        f = band_pass(img, s, e)
        push!(filtered, do_resize(f))
    end
    reference = filtered[1]
    movings = filtered[2:end]
    
    # patch_size=max.(size(fixed) .÷ 8,5)

    rigidly_aligned, params = fourier_mellin_align(reference, movings)
    warps = find_deformations(rigidly_aligned[1], rigidly_aligned[2:4], grid_size, pre_transform_params=params)
    warped = [replace_nan(warp(do_resize(img_list[n+1]), warps[n])) for n=1:length(img_list)-1]
    all_aligned = [do_resize(img_list[1]), warped...]
    return all_aligned, warps #   
end
