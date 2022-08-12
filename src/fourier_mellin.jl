function even_size(sz)
    sz.÷2 .*2
end

# using FourierTools, TestImages, NDTools, View5D, IndexFunArrays
#using SubpixelRegistration
function make_even(img)
    select_region(img, new_size=even_size(size(img)))
end

"""
    fourier_mellin_align(fixed, movings)
alignes a collection of `movings` images to a `fixed` image based on the Fourier-Mellin transformation.
This performs a rigid alignment. The aligned stack including the `fixed` image are returned together with
the tuple (`angles`, `zooms`, `shifts`) as a tuple indicating the alignment parameters.
"""
function fourier_mellin_align(fixed, movings; background=nothing)
    result = [make_even(fixed),]
    # angles = [0.0,]
    # zooms = [1.0,]
    # shifts = [(0.0,0.0),]
    all_params = []
    for moving in movings
        if isnothing(background)
            aligned, params = fourier_mellin(moving, fixed)
        else
            aligned, params = fourier_mellin(moving .- background, fixed.- background)
        end
        push!(result, aligned)
        # push!(angles, α)
        # push!(zooms, azoom)
        # push!(shifts, Tuple(ashift))
        push!(all_params, params)
    end
    return result, all_params
end

function replace_nan(v::AbstractArray)
    map(x -> isnan(x) ? zero(x) : x, v)
end

"""
    fourier_mellin(img1, img2; radial_exponent=2.0, subsampling = 4)
determines scaling, rotation and shift with subpixel precision.
Note that rotation has to be less than +/- 90 degrees to avoid an abiguity in the interpretation, since the
absolute Fourier-transformations, on which the Fourier-Melling transformation is based on is ambigous.
The tuple (`α`, `zoom`, `ashift`) and the transformed aligned `img1` are returned.

#Arguments
+ `img1`:  source image to find transformation for when rotated, zoomed and shifted to the position of `img2`
+ `img2`:  destination image to which `img1` gets aligned to
+ `radial_exponent`:  determines the weighting of the alignment in dependence on the radial frequency.
+ `subsampling`: Determines how coarse or fine the Melling transform, which is based on `nfft_nd` from the FourierTools.jl package, is sampled.
"""
function fourier_mellin(img1, img2; radial_exponent=2.0, subsampling = 4)
    ms = maximum(size(img1))
    szt = (ms, ms) .÷ subsampling

    # the factor below determines the  radius weighting
    period = 180 #
    fm_map(αr) = (0.5*exp(radial_exponent * (αr[2] - 0.5)) * cosd(period*αr[1]), 0.5 * exp(radial_exponent * (αr[2] - 0.5)) * sind(period*αr[1]))

    img1 = make_even(img1)
    img2 = make_even(img2)
    f1 = abs.(nfft_nd(img1, fm_map, szt)) 
    f2 = abs.(nfft_nd(img2, fm_map, szt)) 
    # f_one = abs.(nfft_nd(ones(size(img1)), fm_map, szt)) 

    # sfac = log(0.5^2) - log(2/szt[1]^2)
    # fm_map(t) = (atan(t[2], t[1]) ./ (2π), -0.5 + (log(t[1]^2+t[2]^2 + 0.000001)  - log(2/szt[1]^2)) / sfac)
    # f1 = abs.(nfft_nd(img1, fm_map, szt, is_adjoint=true)) # .* (yy(szt, offset=CtrCorner).+1)
    # f2 = abs.(nfft_nd(img2, fm_map, szt, is_adjoint=true)) # .* (yy(szt, offset=CtrCorner).+1)

    f1 ./= sum(f1) # .* (f_one .+ maximum(f_one)./5)
    f2 ./= sum(f2) # .* (f_one .+ maximum(f_one)./5)

    # @vt f1 f2
    # @show ashift, err, phasediff = phase_offset(f2, f1; upsample_factor=100)
    ashift = find_shift_iter(f2, f1)

    α = -period*pi/180 * ashift[1]/szt[1]
    zoom = ashift[2]/szt[2]
    zoom = exp(radial_exponent * zoom)
    # reg = register(f2, f1; upsample_factor=100)

    # res = rotate(img1, -α)

    # rotα = t -> (cos(α)*t[1] - sin(α)*t[2], sin(α)*t[1] + cos(α)*t[2])
    # res = resample_nfft(img1, t -> rotα(t) .* zoom)

    ϕ = get_rigid_warp((α, zoom, [0.0, 0.0]), size(img1))
    res = replace_nan(warp(img1, ϕ, size(img1)))
    
    # ashift, err, phasediff = phase_offset(res, img2; upsample_factor=100)
    ashift = find_shift_iter(res, img2) # do not normalize, as this may cause trouble, if the mean is close to zero!
    # println("Angle: $α , zoom: $zoom, shift: $ashift") 
    # @vt img2 res
    # ashift = Tuple(ashift ./ size(img1))
    # println("Angle: $α , zoom: $zoom, shift: $ashift") 
    # res2 = resample_nfft(img1, (t) -> rotα(t) .* zoom .+ rotα(ashift).*zoom)

    # @vt img2 res register(img2, res; upsample_factor=100) res2 

    # return res2,(α, zoom, ashift)
    return shift(res, .-ashift), (α, zoom, ashift) # shift(res, -ashift)
end

function test_fm()
    img1 = Float32.(testimage("resolution"));
    img2 = shift(resample_nfft(rotate(img1, 72.0), t->1.7 .*t), (10.2,52.3));

    img3, param = fourier_mellin(img2, img1)
    # img4, param = fourier_mellin(img3, img1)
    # @vt img1 img2, img3

    nimg1 = poisson(img1, 10)
    nimg2 = poisson(img2, 10)

    nimg3, param = fourier_mellin(nimg2, nimg1)
    # @vt nimg1 nimg2, nimg3

end
