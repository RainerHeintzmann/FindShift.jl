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

function inpaint_imgs(imgs, masks)
    if (isnothing(masks))
        return imgs
    end
    inpainted = [];
    for ind in CartesianIndices(imgs)
        if eltype(masks) == Any || isa(eltype(masks), Array)
            push!(inpainted, inpaint(imgs[ind], masks[ind]))
        else
            push!(inpainted, inpaint(imgs[ind],masks))
        end
    end
    return reshape(inpainted, size(imgs))
end


"""
    find_rel_pos(imgs, shift_idx, masks=nothing; sigma = (1,1), shift_vec=nothing, est_std=10.0)

determines the relative shifts in an matrix of images topologically arranged correctly. Note that no further preprocessing or
inpainting is performed on the `imgs`. This all needs to be performed before submitting the images to this routine.

# Arguments
- `imgs`:   the images arranged in a matrix (or higher order tensor)
- 'shift_idx':  the relative topological shift direction to look into in this round
- `masks`:  a single mask (to be applied to all images) or a similarly topologically arranged series of masks.

returns a tensor of relative shifts.
"""
function find_rel_pos(imgs, shift_idx, masks=nothing; sigma = (1,1), shift_vec=nothing, est_std=10.0, exclude_zero=true)
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
                find_shift(ref_img, dat_img; mask1=mask_ref, mask2=mask_dat, exclude_zero=exclude_zero)
            else
                find_shift(ref_img, dat_img; mask1=mask_ref, mask2=mask_dat, est_pos=shift_vec, est_std=est_std, exclude_zero=exclude_zero)
            end
        end 
        # print("idx $(ref_ind), found $(Δx)\n");
        shift_matrix[ref_ind] = Δx[1:2];
    # @vt cor_ref  cor_dat  shift(cor_dat, Δx)
    end
    return shift_matrix
end



"""
    minimize_distances(shift_right_matrix, shift_down_matrix)

finds the best positions to accomodate the measured data of relative shifts between images.
To this aim a system of equations is contructed linking all measured information into a global
minimization problem, which can be solved directly via linear algebra.
M x = C, with C being constants obtained from the measurement and the matrix M indicating which
of the neighboring elements of the 2D measurement are connected via the equations.
The first index of x will be forced to zero.
"""
function minimize_distances(shift_down_matrix, shift_right_matrix)
    # shift_right_matrix = permutedims(shift_right_matrix)
    # shift_down_matrix = permutedims(shift_down_matrix)
    sz2d = max.(size(shift_right_matrix), size(shift_down_matrix))
    pos = zeros(sz2d...,2) # allocated as 3d object but used as 1d object in the linear equation system. Last dimension: x and y
    C = zeros(sz2d...,2);  # allocated as 3D object but used as 1d object in the linear equation system. Last dimension: x and y
    sz3d = size(pos)
    lsz = prod(sz3d)
    M = zeros((lsz, lsz)); # allocated and used as 2d matrix
    # for ind3d in CartesianIndices(sz3d)
    for indxy in CartesianIndices(sz2d)
        ind2d = (indxy[1], indxy[2])
        for ind_vec in 1:sz3d[end] # iterate over vector components
            ind3d = (Tuple(indxy)..., ind_vec);
            ind1d = LinearIndices(pos)[ind3d...]
            ind1d_down = get_lin_idx(pos, ind3d, (1, 0))
            ind1d_up = get_lin_idx(pos, ind3d, (-1, 0))
            if (ind1d_down>0)
                down_dist = shift_down_matrix[ind2d...];
                if !isnan(down_dist[ind_vec])
                    M[ind1d, ind1d] += 1
                    M[ind1d_down, ind1d] += -1
                    C[ind3d...] += -down_dist[ind_vec];
                end
            end
            if (ind1d_up>0)
                up_dist = shift_down_matrix[ind2d[1]-1, ind2d[2]];
                if !isnan(up_dist[ind_vec])
                    M[ind1d, ind1d] += 1
                    M[ind1d_up, ind1d] += -1
                    C[ind3d...] += up_dist[ind_vec];
                end
            end
            ind1d_right = get_lin_idx(pos, ind3d, (0, 1))
            ind1d_left = get_lin_idx(pos, ind3d, (0, -1))
            if (ind1d_right>0)
                right_dist = shift_right_matrix[ind2d...];
                if !isnan(right_dist[ind_vec])
                    M[ind1d, ind1d] += 1
                    M[ind1d_right, ind1d] += -1
                    C[ind3d...] += -right_dist[ind_vec];
                end
                # print("$(ind3d) right dist $(right_dist) result $(C[ind3d...])\n")
            end
            if (ind1d_left>0)
                left_dist = shift_right_matrix[ind2d[1], ind2d[2]-1];
                if !isnan(left_dist[ind_vec])
                    M[ind1d, ind1d] += 1
                    M[ind1d_left, ind1d] += -1
                    C[ind3d...] += left_dist[ind_vec];
                end
            end
        end
    end
    # M[1,:] .= 0;
    # M[1,1] = 1; C[1] = 0; # forcing the x-position of the first solution to zero
    # M[1, lsz÷2+1] = 1; C[lsz÷2+1] = 0; # forcing the y-position of the first solution to zero
    # return nothing, M,C;
    U,D,Vt = svd(M)
    nd = ndims(shift_down_matrix)
    if (D[end-nd-1]/D[1] < 1e-5)
        error("Matrix Singular")
    end
    Dinv = D
    Dinv[1:end-2] = 1 ./D[1:end-nd];
    Dinv = diagm(Dinv)
    pinv = Vt*Dinv*adjoint(U)  # Vt is already transposed
    pos[:] .= pinv*C[:];
    return pos,M,C;
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
        strain[ind] = norm((pos[ind[1]+dirvec[1], ind[2]+dirvec[2],:] .- pos[ind[1],ind[2],:] ) .- shift_matrix[ind])
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
    find_pos(imgs, masks=nothing; shift_vecs = nothing, est_std=15.0, offset=(20, 20), verbose=true, StrainThresh=4.0) # m((6,-9),(3, 22))

determines the relative positions in a matrix of images (topologically assumed to be in the correct order)
"""
function find_pos(images, masks=nothing; shift_vecs = nothing, est_std=15.0, offset=(20, 20), StrainThresh=5.0, do_inpaint=false) # m((6,-9),(3, 22))
    if (do_inpaint)
        images = inpaint_imgs(images, masks);
    end
    if isnothing(shift_vecs)
        shift_down_matrix = find_rel_pos(images, (1,0), masks, est_std=est_std)
        shift_right_matrix = find_rel_pos(images, (0,1), masks, est_std=est_std)
    else  # the user defined some preferred shift
        shift_down_matrix = find_rel_pos(images, (1,0), masks, shift_vec = shift_vecs[1], est_std=est_std)
        shift_right_matrix = find_rel_pos(images,(0,1), masks, shift_vec = shift_vecs[2], est_std=est_std)
    end

    pos, M, C = minimize_distances(shift_down_matrix, shift_right_matrix)

    if (StrainThresh > 0)
        bad_down = limit_strain!(pos, shift_down_matrix, StrainThresh)
        bad_right = limit_strain!(pos, shift_right_matrix, StrainThresh)
        if (bad_down + bad_right > 0)
            print("$(bad_down) vertical and $(bad_right) horizontal correlations were unsuccessful trying to rearrange by ignoring those.\n")
            try 
                pos, M, C = minimize_distances(shift_down_matrix, shift_right_matrix)
            catch
                print("Matrix was singular. Ignoring last attempt to kick out vectors.\n")
            end
        end
    end

    pos .+= reorient(offset .- pos[1,1,:], Val(3))    

    # pos = shift_down_matrixpositon
    # p = scatter(legend=false, framestyle=:box, xlabel="x-", ylabel="y-position")
    # for (x, y) in pos
    #     scatter!([x], [y], markersize=10, markercolor=:blue)
    # end
    # display(p)

    pos = collect(pos[x,y,:] for x = 1:size(pos,1), y = 1:size(pos,2)) # convert to the array of vectors format
    return pos;
end


"""
    get_dt_weights(img, thresh)

generates a weight mask for each tile also avoiding zeros in the title
"""
function get_dt_weights(tile, thresh= mean(tile))
    res = zeros(Bool, size(tile))
    res[1,:] .= true; res[end,:] .= true; res[:,1] .= true; res[:,end] .= true;
    #m = (pb .> thresh) .+  0.0;
    res = res .| (tile .< thresh)
    bw = feature_transform(res)        #DT
    return distance_transform(bw)      #DT
end

"""
    stitch(imgs, pos, big_size = nothing)

pos: a matrix of center positions for each tile in the big image
big_size: optionally specifies the size of the result image
"""
function stitch(imgs, pos, big_size = nothing)
    if (size(imgs, 2) < 2)
        imgs = reshape(imgs, size(pos));
    end
    img_sz = size(imgs[1])
    if (isnothing(big_size))
        max_shift = (0,0)
        for cpos in pos
            max_shift = max.(max_shift, ceil.(Int, abs.(cpos)))
        end
        big_size=(2*max_shift[1]+size(imgs[1],1), 2*max_shift[2]+size(imgs[1],2), img_sz[3]);
    end
    res = zeros(Float64, big_size)  # Specify the type explicitly
    weights = zeros(Float64, big_size)  # Specify the type explicitly
    #wa = zeros(Float64, (160, 160))
    wa = zeros(Float64, (big_size..., img_sz[3]))   #3D
    big_mid = big_size .÷ 2 .+1;
    for (tile, current_pos) in zip(imgs, pos)
        myweights = get_dt_weights(sum(tile, dims=(3,)))[:,:,1];
        # mid_z = size(tile,3)÷2+1;
        cp = (round.(Int, current_pos)...,)
        # print("pos: $(current_pos), cp $(cp) \n")
        res_view = select_region_view(res, new_size=size(tile), center=big_mid .+ (cp..., 0));
        weights_view = select_region_view(myweights, new_size=size(tile)[1:2], center=cp)
        res_view .+= tile .* myweights;
        weights_view .+= myweights;

        # res .+= shift(select_region(weighted, new_size=big_size), (current_pos...,mid_z));  #3D
        # weights .+= shift(select_region(myweights, new_size=big_size[1:2]), (current_pos...,mid_z));
    end
    weights[weights .< 0.001] .= 1e10;
    return res./weights;
end
