# This file contains stitching-related code for FindShift.jl
"""
    in_mask(fkt, data, mask)

applies as reduce function `fkt` only to the `mask`ed area of `data`.
"""
function in_mask(fkt, data, mask)
    fkt(data[mask])
end

"""
    inpaint(img::AbstractArray{T, N}, mask::AbstractArray{TM, N}, border=0.3, mykernel=nothing, usepixels=2) where {T, TM, N}
    
Inpaints areas between the masked areas by using the values of `usepixels` around the borders of the masked areas.
See also the `damp_edge_outside` function in the `FourierTools` toolbox.

# Arguments:
+ `img`: image to inpaint
+ `mask`: (binary) mask to extrapolate from
+ `border`: defines the decay of the inpainting mask kernel (if not provided explicitely). Note that no actual border padding is performed here.
+ `mykernel`: A kernel can be provided. Default: 1/r^3 kernel of support `norm(border size) * sqrt(2)`
+ `usepixels`: number of border pixels in the image to use on each edge. Default: 2. Zero means use all pixels

# Example:
```julia
julia> using TestImages, FindShift, Colors

julia> img = permutedims(Float32.(testimage("fabio_gray_512.png")),[2 1]);

julia> img_d = inpaint(img, img .> 0.5);

julia> Gray.(img_d) 
```
"""
function inpaint(img::AbstractArray{T, N}, mask::AbstractArray{TM, N}; border=0.3, mykernel=nothing, usepixels=2, kernelpower=4) where {T, TM, N}
    rborder=ceil.(Int,border.*size(img));
    new_size=size(img);
    if isnothing(mykernel)
        mykernel = (one(T)./rr(new_size)) .- one(T)./norm(rborder.*sqrt(2.0));
        # clip at zero
        mykernel .= max.(mykernel, zero(T)); 
        mykernel=mykernel .^ kernelpower;
        midpos = size(mykernel) .÷2 .+ 1;
        # This does not matter as it only applies to points directly having data.
        mykernel[midpos...] = one(T);
    end
    transfer=rft(mykernel);
    # weight image
    wimg = (distance_transform(feature_transform(.~mask)) .<= usepixels) .* mask;
    nimg = wimg .* img;

    # should this only work for 2D?
    nimg2 = real(irft(rft(nimg) .* transfer, size(nimg,1))); 
    wimg2 = real(irft(rft(wimg) .* transfer, size(nimg,1)));

    nimg = nimg2 ./ wimg2;
    # replace the original in the middle
    nimg[mask] .= img[mask];
    return nimg;

    # ToDo: Phase subpixel peak determination
    # ToDo: DampEdge (with Gaussian filter)
    # ToDo: make it work in ND 
end

function is_mask_array(masks)
    return !isnothing(masks) && (eltype(masks) == Any || isa(eltype(masks), Array))
end

function inpaint_imgs(imgs, masks)
    if (isnothing(masks))
        return imgs
    end
    inpainted = [];
    for ind in CartesianIndices(imgs)
        if (is_mask_array(masks))
            push!(inpainted, inpaint(imgs[ind], masks[ind]))
        else
            push!(inpainted, inpaint(imgs[ind],masks))
        end
    end
    return reshape(inpainted, size(imgs))
end


"""
    find_rel_pos(imgs, shift_idx, masks=nothing; shift_vec=nothing, est_std=10.0, exclude_zero=true)

determines the relative shifts in an matrix of images topologically arranged correctly. Note that no further preprocessing or
inpainting is performed on the `imgs`. This all needs to be performed before submitting the images to this routine.

# Arguments
- `imgs`:   the images arranged in a matrix (or higher order tensor)
- 'shift_idx':  the relative topological shift direction to look into in this round
- `masks`:  a single mask (to be applied to all images) or a similarly topologically arranged series of masks.
- `shift_vec`:  a vector of shifts to start the search from. If not provided, all results are acceptable. Default: nothing
- `est_std`:  the standard deviation of the Gaussian used for masking the cross-correlation (only if `shift_vec` is provided). Default: 10.0
- `exclude_zero`:  if true, the zero shift will be excluded from the search. Default: true

returns a tensor of relative shifts.
"""
function find_rel_pos(imgs, shift_idx, masks=nothing; shift_vec=nothing, est_std=10.0, exclude_zero=true, dims=1:ndims(imgs))
    mat_sz = size(imgs);

    shift_matrix = Array{NTuple{2,Float64}}(undef, mat_sz[1]-shift_idx[1], mat_sz[2]-shift_idx[2]) 
    for ref_ind in CartesianIndices(size(shift_matrix))
        # print("processing $(ref_ind) \n")
        ref_img = imgs[ref_ind];
        dat_idx = Tuple(ref_ind) .+ shift_idx
        dat_img = imgs[dat_idx...];
        mask_ref, mask_dat = let
            if (!isnothing(masks))
                if eltype(masks) == Any || isa(eltype(masks), Array)
                    masks[ref_ind], masks[dat_idx...]
                else
                    masks, masks;  # use the same mask for all
                end
            else
                nothing, nothing
            end
        end
        # cor_ref = ref_img .- bg_ref; # use_ref
        # cor_dat = dat_img .- bg_dat; # use_dat
        # mask_both = mask_ref # shift(mask_dat.*1.0, shift_vecs[1]) .* mask_ref .> 0.5
        # shift_vec = expand_size(shift_vec, Tuple(zeros(Int, ndims(ref_img))))
        Δx = let 
            if (isnothing(shift_vec))
                find_shift(ref_img, dat_img; mask1=mask_ref, mask2=mask_dat, exclude_zero=exclude_zero, dims = dims)
            else
                find_shift(ref_img, dat_img; mask1=mask_ref, mask2=mask_dat, est_pos=shift_vec, est_std=est_std, exclude_zero=exclude_zero, dims = dims)
            end
        end 
        # print("idx $(ref_ind), found $(Δx)\n");
        shift_matrix[ref_ind] = Δx[1:2];
    # @vt cor_ref  cor_dat  shift(cor_dat, Δx)
    end
    return shift_matrix
end



"""
    minimize_distances(shift_matrices, current_dim)

finds the best positions to accomodate the measured data of relative shifts between images.
To this aim a system of equations is contructed linking all measured information into a global
minimization problem, which can be solved directly via linear algebra.
M x = C, with C being constants obtained from the measurement and the matrix M indicating which
of the neighboring elements of the 2D measurement are connected via the equations.
The first index of x will be forced to zero.

Note that this problem is separable in the dimensions, which is why it should be called for every dimension `current_dim` independently
"""
function minimize_distances(shift_matrices, current_dim)
    # shift_right_matrix = permutedims(shift_right_matrix)
    # shift_down_matrix = permutedims(shift_down_matrix)
    sz2d = max.(size.(shift_matrices)...)
    pos = zeros(sz2d...) # allocated as 3d object but used as 1d object in the linear equation system. Last dimension: x and y
    C = zeros(sz2d...);  # allocated as 3D object but used as 1d object in the linear equation system. Last dimension: x and y
    # sz3d = size(pos)
    lsz = prod(sz2d)
    M = zeros((lsz, lsz)); # allocated and used as 2d matrix
    nd = length(shift_matrices)
    # for ind3d in CartesianIndices(sz3d)
    for indxy in CartesianIndices(sz2d)
        for current_diff_dim in 1:nd # iterate over the various topological difference-directions characterised by the shift_matrices 
            # ind3d = (Tuple(indxy)..., ind_vec);
            ind1d = LinearIndices(pos)[indxy]
            direction_idx = direction_tuple(current_diff_dim, nd)
            ind1d_down = get_lin_idx(pos, indxy, direction_idx)
            if (ind1d_down>0)
                down_dist = shift_matrices[current_diff_dim][indxy];
                if !isnan(down_dist[current_dim])
                    M[ind1d, ind1d] += 1
                    M[ind1d_down, ind1d] += -1
                    C[indxy] += -down_dist[current_dim];
                end
            end
            ind1d_up = get_lin_idx(pos, indxy, .-direction_idx)
            if (ind1d_up>0)
                up_dist = shift_matrices[current_diff_dim][(Tuple(indxy) .- direction_idx)...];
                if !isnan(up_dist[current_dim])
                    M[ind1d, ind1d] += 1
                    M[ind1d_up, ind1d] += -1
                    C[indxy] += up_dist[current_dim];
                end
            end
        end
    end
    # M[1,:] .= 0;
    # M[1,1] = 1; C[1] = 0; # forcing the x-position of the first solution to zero
    # M[1, lsz÷2+1] = 1; C[lsz÷2+1] = 0; # forcing the y-position of the first solution to zero
    # return nothing, M,C;
    U,D,Vt = svd(M)
    if (D[end-nd-1]/D[1] < 1e-5)
        error("Matrix Singular")
    end
    Dinv = D
    Dinv[1:end-1] = 1 ./D[1:end-1];  # there should be exactly one underdetermined direction
    Dinv = diagm(Dinv)
    pinv = Vt*Dinv*adjoint(U)  # Vt is already transposed
    pos[:] .= pinv*C[:];
    return pos;
end

function minimize_distances(shift_matrices)
    sz2d = max.(size.(shift_matrices)...)
    nd = length(shift_matrices)
    pos = zeros(sz2d..., nd) # allocated as 3d object but used as 1d object in the linear equation system. Last dimension: x and y

    for current_dim = 1:nd
        pos[:,:,current_dim] = minimize_distances(shift_matrices, current_dim)
    end
    return pos
end

"""
    get_strain(pos, shift_matrix)

determines the energy which each relative shift is experiencing with the determined positions `pos` and the given `shift_matrix`.
The shift direction is determined from the size of the pos and shift_matrix.
"""
function get_strain(pos, shift_matrix)
    dirvec = size(pos)[1:end-1] .- size(shift_matrix);
    strain = zeros(size(shift_matrix));
    for ind in CartesianIndices(shift_matrix)
        ind_shift = Tuple(ind) .+ dirvec 
        strain[ind] = norm((pos[ind_shift..., :] .- pos[Tuple(ind)..., :] ) .- shift_matrix[ind])
    end
    return strain
end

"""
    limit_strain!(pos, shift_matrix, StrainThresh=4.0)

limits the strain if a given `shift_matrix` and consolidated positions `pos` by dropping shifts from the matrix (i.e. setting them to NaN).
"""
function limit_strain!(pos, shift_matrix, StrainThresh=4.0)
    strain = get_strain(pos, shift_matrix)
    N = 0;
    for ind in CartesianIndices(shift_matrix)
        if (strain[ind] .> StrainThresh)
            shift_matrix[ind] = (NaN, NaN);
            N += 1;
        end
    end
    return N;
end


"""
    find_pos(imgs, masks=nothing; shift_vecs = nothing, est_std=15.0, offset=(20, 20), verbose=true, StrainThresh=4.0, dims=1:ndims(images)) # m((6,-9),(3, 22))

determines the relative positions in a matrix of images (topologically assumed to be in the correct order)
"""
function find_pos(images, masks=nothing; shift_vecs = nothing, est_std=10.0, offset=(20, 20), StrainThresh=5.0, do_inpaint=false, dims=1:ndims(images)) # m((6,-9),(3, 22))
    if (do_inpaint)
        images = inpaint_imgs(images, masks);
    end
    nd = ndims(images)
    shift_matrices = []
    for d=1:nd
        current_diff_dir = direction_tuple(d, nd)
        if isnothing(shift_vecs)
            push!(shift_matrices, find_rel_pos(images, current_diff_dir, masks, est_std=est_std, dims=dims))
        else  # the user defined some preferred shift
            push!(shift_matrices, find_rel_pos(images, current_diff_dir, masks, shift_vec = shift_vecs[d], est_std=est_std, dims=dims))
        end
    end
    pos = minimize_distances(shift_matrices)

    if (StrainThresh > 0)
        bad_down = limit_strain!(pos, shift_matrices[1], StrainThresh)
        bad_right = limit_strain!(pos, shift_matrices[2], StrainThresh)
        if (bad_down + bad_right > 0)
            print("$(bad_down) vertical and $(bad_right) horizontal correlations were unsuccessful trying to rearrange by ignoring those.\n")
            try 
                pos = minimize_distances(shift_matrices)
            catch
                print("Matrix was singular. Ignoring last attempt to kick out vectors.\n")
            end
        end
    end

    pos .+= reorient(offset .- pos[ones(Integer, ndims(pos)-1)..., :], Val(ndims(pos)))

    # pos = shift_down_matrixpositon
    # p = scatter(legend=false, framestyle=:box, xlabel="x-", ylabel="y-position")
    # for (x, y) in pos
    #     scatter!([x], [y], markersize=10, markercolor=:blue)
    # end
    # display(p)

    pos = collect(Tuple(pos[Tuple(ind)...,:]) for ind = CartesianIndices(size(pos)[1:end-1])) # convert to the array of vectors format
    return pos;
end


"""
    get_dt_weights(img, thresh)

generates a weight mask for each tile also avoiding zeros in the title
"""
function get_dt_weights(mask, blend_border = 0.5)
    bw = feature_transform(.~mask)        #DT
    dt = distance_transform(bw)
    dt = max.(one(eltype(dt)) .- dt ./ (maximum(dt) * blend_border), zero(eltype(dt)))
    return (one(eltype(mask)).+cos.(π*dt))/2
end

"""
    stitch(imgs, pos, big_size = nothing)

pos: a matrix of center positions for each tile in the big stitched image
big_size: optionally specifies the wanted size of the result image

The positions can be obtained by using `find_pos` on the stack of images (including possible masks).

"""
function stitch(imgs, pos, masks=nothing; big_size = nothing, blend_border = 0.5, Eps = 0.001, diagnostic=false)
    if (ndims(imgs) < ndims(pos))
        imgs = reshape(imgs, size(pos));
    end
    # nd = ndims(imgs[1])
    img_sz = size(imgs[1]) # size of one individual image
    DT = eltype(imgs[1])
    if (isnothing(big_size))
        max_shift = zeros(Int, ndims(pos))
        for cpos in pos
            max_shift = max.(max_shift, ceil.(Int, abs.(cpos)))
        end
        big_size = expand_size(Tuple(2 .*max_shift .+ img_sz[1:length(max_shift)]), img_sz)
    end
    res = zeros(DT, big_size)  # Specify the type explicitly

    res_full = nothing
    if (diagnostic)
        res_full = zeros(DT, (big_size..., prod(size(imgs))))  # make space for individual images
    end
    #wa = zeros(Float64, (160, 160))
    # wa = zeros(Float64, (big_size..., img_sz[3]))   #3D
    # wa = zeros(Float64, big_size)   #3D
    big_mid = big_size .÷ 2 .+1;    
    myweights, weight_size = let
        if (isnothing(masks))
            mask = ones(Bool, img_sz[1:length(pos[1])])
            set_border!(mask, false)
            get_dt_weights(mask, blend_border), big_size[1:ndims(mask)]
        else
            if (!is_mask_array(masks))
                mask = copy(masks)
                set_border!(mask, false)
                get_dt_weights(mask, blend_border), big_size[1:ndims(mask)] # creates a distance-based weight to the border            
            else
                nothing, big_size[1:ndims(masks[1])]
            end
        end
    end
    weights = zeros(eltype(imgs[1]), weight_size)  # Specify the type explicitly

    tn=1
    for (tile, current_pos, tile_idx) in zip(imgs, pos, CartesianIndices(size(imgs)))            
        myweights = let
        if (is_mask_array(masks))
            mask = copy(masks[tile_idx])
            set_border!(mask, false)
            get_dt_weights(mask, blend_border);
        else
            myweights
        end
        end
        # mid_z = size(tile,3)÷2+1;
        cpos = (round.(Int, current_pos)...,)
        # print("pos: $(current_pos), cp $(cp) \n")
        res_view = select_region_view(res, new_size=size(tile), center = expand_size(cpos .+ big_mid[1:length(cpos)], big_mid));
        weights_view = select_region_view(weights, new_size=size(myweights), center = cpos .+ big_mid[1:length(cpos)])
        if (diagnostic)
            res_full_view = select_region_view(res_full, new_size=(size(tile)...,1), center = (expand_size(cpos .+ big_mid[1:length(cpos)], big_mid)..., tn));
            res_full_view .+= tile;
        end
        res_view .+= tile .* myweights;
        weights_view .+= myweights;
        tn += 1;

        # res .+= shift(select_region(weighted, new_size=big_size), (current_pos...,mid_z));  #3D
        # weights .+= shift(select_region(myweights, new_size=big_size[1:2]), (current_pos...,mid_z));
    end
    weights[weights .< Eps] .= Eps;
    if (diagnostic)
        return res./weights, res_full
    end
    return res./weights;
end
