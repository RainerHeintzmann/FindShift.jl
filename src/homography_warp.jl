function idx_to_coord(idx, sz, edge_frac=1/7)
    round(Int, (idx==1) ?  sz.*edge_frac .+1 : sz.*(1 .- edge_frac) .+1)
end
"""
    get_default_markers(image)
    returns the default marker positions of a given `image` dataset.
    The default is at 25% and 75% along each dimension.
"""
function get_default_markers(image)
    NIdx = Tuple(2 .*ones(Int, ndims(image)))
    [idx_to_coord.(Tuple(idx), size(image)) for idx in CartesianIndices(NIdx)[:]]
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
function extract_patches(img, markers=nothing, patch_size=max.(size(img) .÷ 5,5))
    patches = []
    if isnothing(markers)
        markers = get_default_markers(img)
    end
    for m in markers
        patch = select_region(img, new_size=patch_size, center=Tuple(m))
        push!(patches, patch)
    end
    return patches, markers # cat(patches..., dims=ndims(img)+1)
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
    compute_homography(markers1, markers2)
    computes the homography matrix H which means
    H [markers1[n]...,1] = [markers2[n]...,1] etc
"""
function compute_homography(markers_from, markers_to; extra_constraint=false)
    # matrix containing the coefficient to look for:
    # 10th eq. for the contraint that the last entry needs to be always one.
    A = zeros(2 * length(markers_from) + 2*extra_constraint, 9) #  + 3
    index = 1
    for (match1, match2) in zip(markers_from, markers_to)
        base_index_x = index * 2 - 1
        base_index_y = 1:3
        A[base_index_x, base_index_y] = Float64.([match1...; 1.0;])
        A[base_index_x + 1, 4:6] = A[base_index_x, base_index_y]
        A[base_index_x, 7:9] =
            -1.0 * A[base_index_x, base_index_y] * match2[1]
        A[base_index_x + 1, 7:9] =
            -1.0 * A[base_index_x, base_index_y] * match2[2]
        index += 1 
    end
    enhancement = 1e8
    # A[9, 9] = enhancement*one(eltype(A))  # enforce the result vector to be normalized
    if extra_constraint
        A[9, 8] = enhancement*one(eltype(A))  # enforce the result vector to be normalized
        A[10, 7] = enhancement*one(eltype(A))  # enforce the result vector to be normalized
    end
    # b = eltype(A).([0,0,0,0,0,0,0,0,enhancement,0,0])
    # # b = eltype(A).([1,1,1,1,1,1,1,1,1])
    # ns = A\b
    # return SMatrix{3,3}(reshape(ns, (3, 3))') # the normalization needs to be taken care of 
  
    if extra_constraint
        ns = nullspace(A, rtol=1e-9)
    else
        ns = nullspace(A)
    end
    return SMatrix{3,3}(reshape(ns, (3, 3))') / ns[end] # the normalization needs to be taken care of 
    # return SMatrix{3,3}(reshape(ns ./ ns[end], (3, 3))')  # bad results

end

function get_warp(H::SArray{Tuple{S, S}, T, N})  where {S,T,N}
    tmp = ones(Float64,S)
    tmp2 = zeros(Float64,S)
    function ϕ(x) # where {S2, T2}
        tmp[1:S-1] .= Float64.(x)
        # return @view (H*tmp)[1:S-1] # ./((H*[x..., one(T)])[S])
        tmp2 .= H*tmp
        tmp2 ./= tmp2[S]
        return @view tmp2[1:S-1] # the @view is essential to not allocate memory
    end
    return ϕ
end

# function get_affine_warp(H::SArray{Tuple{S, S}, T, N})  where {S,T,N}
#     function ϕ(x::SArray{Tuple{S}, Int64, 1, S}) 
#         return H*x # the @view is essential to not allocate memory
#     end
#     return ϕ
# end

"""
    get_homography_warp(markers_from, markers_to)
    determines the homography transform from markers1 to markers2.
    The result is a function the can be applied to an `SVector` and thus
    be directly used in `warp(image, ϕ, axes(image))` to warp the `image`.
"""
function get_homography_warp(markers_from, markers_to, rot_mat=nothing)
    if !isnothing(rot_mat)
        # irot = inv(rot_mat)
        markers_to_2 = [(rot_mat*[m...,1.0])[1:2] for m in markers_to]
        H = compute_homography(markers_from, markers_to_2) 
        w2 = get_warp(H)
        return w2
        # w1 = get_warp(SMatrix{3,3}(rot_mat))
        # return x -> w2(w1(x))
    else
        H = compute_homography(markers_from, markers_to) 
        w2 = get_warp(H)
        return w2
    end
end

function get_shift(myshift)
    SMatrix{3,3}([1.0 0.0 myshift[1];0.0 1.0 myshift[2];0.0 0.0 1.0])
end

"""
    get_rotation(moving, Φ)
    returns a rotation operation in homography coordinates by the angle `Φ` in rad, which accounts for the middle pixel kept unchanged.
"""
function get_rotation(moving, Φ)
    midpos = (size(moving) .÷2) .+1
    shift_mat = get_shift(.-midpos)
    rot_mat = SMatrix{3,3}([cos(Φ) -sin(Φ) 0; sin(Φ) cos(Φ) 0; 0.0 0.0 1.0])
    shift_mat2 = get_shift(midpos) 
    return shift_mat2 * rot_mat * shift_mat #  
end

"""
    determine_homography_warps(fixed, movings, markers=nothing; patches=nothing, max_shift = 300.0, patch_size=max.(size(fixed) .÷ 5,5), pre_rotate_angles=nothing)

determines the homography-based warp between a `fixed` image and a number of `moving` images.
The warps are returned as an iterable and can be used in the `warp` function.
`markers` determine the locations in the `fixed` image at which reference patches are extracted.
Their locations are then (hopefully) retrieved in the `moving` images.

#Arguments
+ fixed:    the reference image to align to
+ moving:   a collection (vector) or images that should be aligned to `fixed`. 
+ markers:  a vector of center positions where the alignment should be determined. If `nothing` is provided, positions at 25% and 75% of each coordinates are chosen.

#Example:
    markers = [[100,100],[1000,100],[1000,1000,],[100,1000]]
    w1,w2,w3 = determine_homography_warp(fixed, [moving1, moving2, moving3], markers)
"""
function determine_homography_warps(fixed, movings, markers=nothing; patches=nothing, max_shift = 300.0, patch_size=max.(size(fixed) .÷ 5,5), pre_rotate_angles=nothing)
    q, markers = extract_patches(fixed, markers, patch_size)
    ws = []
    num = 1
    if !isnothing(patches)
        while !isempty(patches)
            pop!(patches)
        end
    end
    for moving in movings
        rot_mat = nothing
        if !isnothing(pre_rotate_angles) && !isnothing(pre_rotate_angles[num])
            Φ = pre_rotate_angles[num]*pi/180 # converted to rad
            moving = FourierTools.rotate(moving, Φ) # keeps the (Fourier-type) center pixel constant
            rot_mat = get_rotation(moving, Φ)
        end
        located_pos, shifted = locate_patches(q, moving)
        @show located_pos
        if !isnothing(patches)
            push!(patches, shifted)
        end
        pos=1
        for (p,r) in zip(located_pos, markers)
            dist = norm(p .- r)
            if  dist .> max_shift
                @warn("A located position $(pos) in image $(num) is at distance $(dist), too far way from the reference position.")
                # @vt moving shifted
                # error("inacceptably large displacement bailing out.")
            end
            pos += 1
        end
        push!(ws, get_homography_warp(markers, located_pos, rot_mat))
        num += 1
    end
    return ws
end
