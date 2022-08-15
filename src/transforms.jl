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

