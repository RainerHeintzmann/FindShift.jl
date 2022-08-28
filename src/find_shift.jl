# add FourierTools,IndexFunArrays, NDTools, Optim, Zygote, LinearAlgebra, ChainRulesCore
# add View5D, TestImages, Noise, FiniteDifferences

"""
    FindMethod

Abstract supertype for all FindMethod types.

See [`FindZoomFT`](@ref), [`FindIterative`](@ref).
"""
abstract type FindMethod end 

"""
FindZoomFT

Uses a Zoomed FT to localize the peak to sub-pixel precision.
"""
struct FindZoomFT <: FindMethod end

"""
FindIter

Uses an iterative peak optimization to localize the peak to sub-pixel precision.
"""
struct FindIter <: FindMethod end

function dist_sqr(dat1, dat2)
    sum(abs2.(dat1 .- dat2))
end

function dist_anscombe(dat1, dat2)
    sum(abs2.(sqrt.(dat1) .- sqrt.(dat2)))
end

"""
    find_ft_shift_iter(fdat1, fdat2, Δx=nothing; max_range=nothing, verbose=false, mynorm=dist_sqr)

finds the shift between two input images by minimizing the distance.
To be fast, this distance is calculated in Fourierspace using Parseval's theorem.
Therefore `fdat1` and `fdat2` have to be the FFTs of the data. 
Returned is an estimate of the real space (subpixel) shift.

The first image is interpreted as the ground truth, whereas the second as a measurement
described by the distance norm `mynorm`. 
returned is the shift vector.
"""
function find_ft_shift_iter(fdat1::AbstractArray{T,N}, fdat2::AbstractArray{T,N}; max_range=nothing, verbose=false, mynorm=dist_sqr, normalize_variance=true) where {T,N}
    # loss(v) = mynorm(fdat1, v[1].*exp_shift_dat(fdat2,v[2:end])) 
    # win = window_hanning(size(fdat2), border_in=0.0, border_out=1.0)  # window in frequency space
    if normalize_variance
        v1 = sum(fdat1.*conj(fdat1))
        v2 = sum(fdat2.*conj(fdat2))
        fdat1 = fdat1./sqrt(v1) # to help the optimization
        fdat2 = fdat2./sqrt(v2) # to help the optimization
    end
    init_loss = mynorm(fdat1, fdat2)
    if init_loss != 0
        fdat1 = fdat1./sqrt(init_loss) # to help the optimization
        fdat2 = fdat2./sqrt(init_loss) # to help the optimization
    end

    loss(v) = mynorm(fdat1, exp_shift_dat(fdat2, v)) # win .* 
    function g!(G, x)  # (G, x)
        G .= gradient(loss, x)[1]
    end
    # p_est = [1.0, zeros(ndims(fdat2))...]
    p_est = zeros(real(T), ndims(fdat2))
    od = OnceDifferentiable(loss, g!, p_est)
    res = let
        if !isnothing(max_range)
            lower = k_est .- max_range
            upper = k_est .+ max_range
            optimize(od, lower, upper, p_est, Fminbox(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) # {GradientDescent}, , x_tol = 1e-2, g_tol=1e-3
        else
            optimize(od, p_est, LBFGS(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
        end
    end
    # res.minimizer[2:end], res.minimizer[1]
    # return the result modulo the image size. Sometimes the optimization finds a multiple of this.
    pos_sign = (res.minimizer .>= 0).*2 .-1 # a sign that cannot become zero
    return mod.(res.minimizer, pos_sign .* size(fdat1))
end

# using View5D
"""
    find_shift_iter(dat1, dat2, Δx=nothing)

finds the integer shift with between two input images via the maximum of the cross-correlation.
Therefore `dat1` and `dat2` have to the the FFTs of the data. 
Returned is an estimate of the real space integer pixel shift.

The first image is interpreted as the ground truth, whereas the second as a measurement.
Returned is the shift vector.
"""
function find_shift(dat1, dat2)
    mycor = let
        if all((size(dat2).-size(dat1)) .== 0)
            if eltype(dat1)<:Real && eltype(dat2)<:Real
                fftshift(irfft(rfft(dat1) .* conj(rfft(dat2)), size(dat1)[1]))
            else
                fftshift(ifft(fft(dat1) .* conj(fft(dat2))))
            end
        else # normalize the correlation appropriately to identify the snippet
            dat2_big = select_region(dat2, new_size=size(dat1))
            # ref = select_region(ones(eltype(dat1), size(dat2)), new_size=size(dat1))
            if eltype(dat1)<:Real && eltype(dat2)<:Real
                rft1 = rfft(dat1)
                cor1 = fftshift(irfft(rft1 .* conj(rfft(dat2_big)), size(dat1)[1]))
                # cor2 = fftshift(irfft(rft1 .* conj(rfft(ref)), size(dat1)[1]))
                # res = cor1 ./ (abs.(cor2) .+ maximum(cor1) ./ 100)
                # @vt cor1 res
            else
                rft1 = fft(dat1)
                cor1 = fftshift(ifft(rft1 .* conj(fft(dat2_big))))
            end
            cor1
        end
    end
    if eltype(dat1)<:Real && eltype(dat2)<:Real
        return find_max(mycor, exclude_zero=false)
    else
        return find_max(abs2.(mycor), exclude_zero=false)
    end
end

"""
    find_shift_iter(dat1, dat2, Δx=nothing)

finds the shift between two input images by minimizing the distance.
To be fast, this distance is calculated in Fourierspace using Parseval's theorem.
Therefore `dat1` and `dat2` have to the the FFTs of the data. 
Returned is an estimate of the real space (subpixel) shift.

The first image is interpreted as the ground truth, whereas the second as a measurement
described by the distance norm `mynorm`.

#Arguments
+ `dat1`:   source dataset for which the shift towards the destination dataset `dat2` is determined
+ `dat2`:   destination dataset towareds which the shift is determined
+ `Δx`:     an initial estimate of the shift (the default value of `nothing` means that this is automatically determined via a cross-correlation)
"""
function find_shift_iter(dat1::AbstractArray{T,N}, dat2::AbstractArray{T,N}, Δx=nothing, normalize_variance=true) where {T,N}# max_range=nothing, verbose=false, mynorm=dist_sqr
    # mycor = irft(rft(dat1) .* conj(rft(dat2)), size(dat1)[1])
    Δx = let
        if isnothing(Δx)
            find_shift(dat1, dat2)
        else
            Δx
        end
    end
    # @show Δx

    dat1c, dat2c = shift_cut(dat1, dat2, .-Δx)
    # return dat1, dat2, mycor, Δx
    win = window_hanning(T, size(dat1c), border_in=0.0)

    # @show any(isnan.(win .* dat1c))
    # @show any(isnan.(win .* dat2c))
    sub = find_ft_shift_iter(ft(win .* dat1c), ft(win .* dat2c), normalize_variance=normalize_variance)
    if norm(sub) > T(2.0) # why is the shift sometimes > 1.0 ? Maybe since the norm and the normalization is different?
        @warn "Problem in maximizing correlation. Using integer maximum instead."
        return Δx
    else
        return Δx .+ sub # , 1 ./scale
    end
end


"""
    shift_cut(dat1, dat2, Δx)

assumes that input image `dat2` is a version of `dat1` but shifted by `Δx` (in the coordinates of dat1).
returned are both images shifted to the same coordinate system and cut such that no wrap-around occurs.
The result snippet will have the size of dat2 which can be smaller than the size of dat1.
"""
function shift_cut(dat1, dat2, Δx)
    sz1 = size(dat1)
    sz2 = size(dat2)
    szn =  min.(sz1 .- abs.(Δx), sz2)
    # ctrn = (szn .÷2) .+ 1
    c1 = (sz1 .÷2).+1 .- Δx # ifelse.(Δx .> 0, ctrn, (2 .* sz1 .- szn) .÷2 .+1)
    c2 = (sz2 .÷2).+1 # ifelse.(Δx .> 0, (2 .* sz2 .- szn) .÷2 .+1, ctrn)
    return select_region(dat1, center = c1, new_size=szn), select_region(dat2, center = c2, new_size=szn)
end

function find_ft_iter(dat::AbstractArray{T,N}, k_est=nothing; exclude_zero=true, max_range=nothing, verbose=false) where{T,N}
    RT = real(T)
    win = collect(window_hanning(RT, size(dat), border_in=0.0))
    wdat = win .* dat 

    k_est = let
        if isnothing(k_est)
            find_max(abs.(ft(wdat)), exclude_zero=exclude_zero)
        else
            k_est
        end
    end
    k_est = RT.([k_est...])
    v_init = sqrt(abs2_ft_peak(win .* dat, k_est))
    if v_init != 0.0
        wdat ./= v_init
    else
        @warn "FT peak is zero."
    end

    mynorm2(x) = -abs2_ft_peak(wdat, x)  # to normalize the data appropriately
    #@show mynorm(k_est)
    #@show typeof(k_est)
    #@show mygrad(k_est)
    function g!(G, x)  # (G, x)
        G .= gradient(mynorm2,x)[1]
    end
    # x = -1:0.1:1; plot(x,[mynorm2(k_est .+ [0,p]) for p in x]); plot!(x,[mynorm2(k_est .+ [p,0]) for p=-1:0.1:1])
    # plot(x,[gradient(mynorm2,(k_est .+ [0,p]))[1][2] for p in x]); plot!(x,[gradient(mynorm2,(k_est .+ [p,0]))[1][1] for p in x])
    # ff(p) = gradient(mynorm2, k_est .+ [p,0])[1][1]
    # ff2(p) = grad(central_fdm(5, 1), mynorm2, k_est .+ [p,0])[1][1]
    # gg(p) = gradient(mynorm2, k_est .+ [0, p])[1][2]
    # gg2(p) = grad(central_fdm(5, 1), mynorm2, k_est .+ [0, p])[1][2]
    # plot(x,[gg(p) for p in x]); plot!(x,[gg2(p) for p in x]); 
    # plot!(x,[ff(p) for p in x]); plot!(x,[ff2(p) for p in x])
    # od = OnceDifferentiable(mynorm2, k_est; autodiff = :forward);
    od = OnceDifferentiable(mynorm2, g!, k_est)
    #@show fieldnames(typeof(od))
    res = let
        if !isnothing(max_range)
            lower = k_est .- max_range
            upper = k_est .+ max_range
            @time optimize(od, lower, upper, k_est, Fminbox(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) # {GradientDescent}, , x_tol = 1e-2, g_tol=1e-3
        else
            @time optimize(od, k_est, LBFGS(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
            # @time optimize(od, k_est, GradientDescent(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
            # @time optimize(mynorm2, g!, k_est, LBFGS(), Optim.Options(show_trace=true)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
        end
    end
    # Optim.minimizer(res), mynorm(Optim.minimizer(res))
    res.minimizer
end

"""
    find_ft_peak(dat, k_est=nothing; method=:FindZoomFT::FindMethod, interactive=true, overwrite=true, exclude_zero=true, max_range=nothing, scale=40, abs_first=true, roi_size=10)

localizes the peak in Fourier-space with sub-pixel accuracy using an iterative fitting routine.

# Arguments
+ dat:  data
+ k_est: a starting value for the peak-position. Needs to be accurate enough such that a gradient-based search can be started with this value
        If `nothing` is provided, an FFT will be performed and a maximum is chosen. 
+ method: defines which method to use for finding the maximum. Current options: `:FindZoomFT` or `:FindIter`
+ interactive: a boolean defining whether to use user-interaction to define the approximate peak position. Only used if `k_est` is `nothing`.
+ ignore_zero: if `true`, the zero position will be ignored. Only used if `!isnothing(k_est)`.
+ max_range: maximal search range for the iterative optimization to be used as a box-constraint for the `Optim` package.
+ overwrite: if `false` the previously saved interactive click position will be used. If `true` the previous value is irnored and the user is asked to click again.
+ max_range: maximal search range to look for in the +/- direction. Default: `nothing`, which does not restrict the local search range.
+ exclude_zero: if `true`, the zero pixel position in Fourier space is excluded, when looking for a global maximum
+ scale: the zoom scale to use for the zoomed FFT, if `method==:FindZoomFT`
"""
function find_ft_peak(dat::AbstractArray{T,N}, k_est=nothing; method=:FindZoomFT, interactive=false, overwrite=true, exclude_zero=true, max_range=nothing, scale=40, abs_first=true, roi_size=10, verbose=false) where{T,N}
    fdat = ft(dat)
    RT = real(T)
    k_est = let
        if isnothing(k_est)
            if interactive
                pos = get_positions(fdat, overwrite=overwrite)
                pos[1] .- (size(dat).÷ 2 .+1)
            else
                RT.(find_max(fdat, exclude_zero=exclude_zero))
            end
        else
            k_est
        end
    end
    k_est = [k_est...]
    if method == :FindZoomFT
        k_est = Tuple(round.(Int, k_est))
        return get_subpixel_peak(fdat, k_est, scale=scale, exclude_zero=exclude_zero, abs_first=abs_first, roi_size=roi_size)
    elseif method == :FindIter
        return find_ft_iter(dat, k_est; exclude_zero=exclude_zero, max_range=max_range, verbose=verbose);
    else
        error("Unknown method for localizing peak. Use :FindZoomFT or :FindIter")
    end
end

"""
    align_stack(dat; refno=1, ref = nothing, damp=0.1, max_freq=0.4,  dim=ndims(dat), method=:FindZoomFT)

aligns a series of images with respect to a reference image `ref`. 
If `ref==nothing` this image is extracted from the stack at position `refno`. If `refno==nothing` the central position is used.
If `eltype(dat)` is an Integer Type, the data will be casted to `Float32`.

# Arguments
+ dat: stack to align slicewise
+ ref: reference image to align to (if `isnothing(ref)==false`)
+ reno: number of reference imag in stack (only if `isnothing(ref)==true`)
+ damp: percentage of outer region to damp
+ max_freq: maximal frequency to consider in the correlation

# Example
```julia
julia> dat = rand(100,100);

julia> N=10; sh = [rand(2) for d=1:N]; sh[N÷2+1] = [0.0,0.0]

julia> dats = cat((shift(dat, .- sh[d]) for d=1:N)..., dims=3);

julia> dat_aligned, shifts = align_stack(dats);

juliat> shifts .- sh 
```
"""
function align_stack(dat::AbstractArray{T,N}; refno=nothing, ref = nothing, damp=0.1, max_freq=0.4, dim=ndims(dat), method=:FindIter, shifts=nothing) where{T,N}
    RT = real(T)
    refno = let
        if isnothing(refno)
            refno=size(dat)[dim] ÷2 +1
        else
            refno
        end
    end
    if eltype(dat) <: Integer
        dat = Float32.(dat)
    end
	ref = let
		if isnothing(ref)
			ref = slice(dat, dim, refno)
		else
			ref
		end
	end
	# damp_edge_outside(ref, damp)
	wh = window_hanning(RT, size(ref), border_in=0.0, border_out=1.0-damp)
	fwin = window_radial_hanning(RT, size(ref), 		 
  		border_in=max_freq*0.8,border_out=max_freq*1.2)
		# rr(size(ref)) .< maxfreq .* size(ref)[1]
	imgs = []
	res_shifts = []
	fref = fwin .* conj(ft(wh .* (ref .- mean(ref))))
	for n=1:size(dat,dim)
        aslice = slice(dat, dim, n)
        # aslice = dropdims(aslice, dims=dim)
        myshift = let 
            if isnothing(shifts)
                cor_ft = ft(wh .* (aslice .- mean(aslice))) .* fref
                find_ft_peak(cor_ft, method=method, interactive=false)
            else
                shifts[n]
            end
        end
		# m, p = findmax(abs.(cor))
		# midp = size(cor).÷ 2 .+1
		# myshift = midp .- Tuple(p)
		push!(res_shifts, myshift)
		push!(imgs, shift(aslice, myshift))
		# push!(imgs, cor_ft)
	end
	cat(imgs...,dims=dim), res_shifts
end


# Lets find the peaks by autocorrelation:
function correl_at(k_cur, dat, other=dat)
    # sum(dat .* conj.(other) .* exp_ikx(size(dat)[1:2], shift_by=k_est), dims=(1,2))
    
    sum(dat .* conj.(other) .* exp_shift(size(dat), k_cur))
    
    # exp_ikx(size(dat)[1:2], shift_by=k_cur),dims=(1,2))
    # ft2d(dat .* conj.(other)) 
end

function all_correl_strengths(k_cur, dat, other=dat)
    sum(abs.(correl_at(k_cur, dat, other)))  # sum of all correlations
end

function optim_correl(dat, other=dat; k_est = nothing, method=:FindZoomFT, verbose=false)
    if true
        find_ft_peak(dat .* conj.(other), method=method, verbose=verbose)
    else # old version below
        mynorm(x) = - all_correl_strengths(x, dat, other)
        #@show k_est
        lower = k_est .- 2.1
        upper = k_est .+ 2.1
        mygrad(x) = gradient(mynorm,x)[1]
        function g!(G, x)
            G .= mygrad(x)
        end
        od = OnceDifferentiable(mynorm, g!, k_est)
        res = optimize(od,lower, upper, k_est, Fminbox()) # {GradientDescent}
        # res = optimize(mynorm, k_est, LBFGS()) #NelderMead()
        Optim.minimizer(res)
    end
end

"""
     prepare_correlation(dat; upsample=true, other=nothing, psf=nothing)

prepares the correlation by optionally convolving `dat` with the `psf` and upsampling the result by a factor of two.
This is needed since the correlation can extend twice as far.
`dat` should already be the inverse Fourier transformation of the data to correlate.
To correlate the Fourier transforms of real-valued data, just supply the real-space data as `dat`.
"""
function prepare_correlation(dat; upsample=true, other=nothing, psf=nothing)
    up = zeros(size(dat,1)*2,size(dat,2)*2,size(dat,3))
    dat = let
    if !isnothing(psf)
        conv_psf(dat, psf)
    else
        dat
    end
    end
    if upsample
    for n=1:size(dat,3)
        up[:,:,n] .= upsample2(dat[:,:,n])
    end
    else
        up = dat
    end
    other = let
        if isnothing(other)
            up
        else
            other = let
                if false # !isnothing(psf)
                    conv_psf(other, psf)
                else
                    other
                end
            end
            up2 = zeros(size(other,1)*2,size(other,2)*2,size(other,3))
            if upsample
                for n=1:size(other,3)
                    up2[:,:,n] .= upsample2(other[:,:,n])
                end
            else
                up2 = other
            end
            up2
        end
    end
    return up, other
end

# note that dat and other are already in Fourier-space
"""
    correlate(dat; upsample=true, other=nothing, phase_only=false, psf=nothing)

correlates `dat` with a reference `other` or by default with itself.
`dat` should already be the inverse Fourier transformation of the array to correlate.
To correlate the Fourier transforms of real-valued data, just supply the real-space data as `dat`.
"""
function correlate(dat; upsample=true, other=nothing, phase_only=false, psf=nothing)
    up, other = prepare_correlation(dat; upsample=upsample, other=other, psf=psf)
    myprod = up .* conj.(other)
    if phase_only
        myprod ./= ifelse.(iszero.(myprod), eltype(myprod)(Inf), (abs.(myprod)))
    end
    mycor = ft2d(myprod)  # correlates starting in Fourier space
    # mycor = ft2d(abs2.(up))  # correlates starting in Fourier space
    return mycor
end

#  conclusion: cc seems to be the most reliable correlation

# now we try to find the subpixel position using a chirped z-transform

function get_subpixel_patch(cor::AbstractArray{T,N}, p_est; scale=10, roi_size=4) where{T,N}
    RT = real(T)
    p_mid = size(cor).÷2 .+ 1
    new_size = min.(roi_size .* scale, size(cor))
    roi = select_region(cor,center=p_mid.+p_est,new_size=new_size)  # (iczt(fc .* exp_ikx(size(cor)[1:2], shift_by=.-p_est), scale)
    # return roi
    fc = ft2d(roi .* window_hanning(RT, size(roi), border_in=0.0))
    # fc = ft2d(roi)
    scale = scale .* ones(RT, ndims(cor))
    roi = iczt(fc, scale, (1,2))
    return roi
end

function get_subpixel_peak(cor::AbstractArray{T,N}, p_est=nothing; exclude_zero=true, scale=10, roi_size=4, abs_first=false, dim=3) where{T,N}
    p_est = let
        if isnothing(p_est)
            find_max(cor, exclude_zero=exclude_zero)
        else
            p_est
        end
    end
    # @show roi_size
    roi = let
        if abs_first 
            get_subpixel_patch(sum(abs.(cor), dims=dim), p_est, scale=scale, roi_size=roi_size)
        else
            sum(abs.(get_subpixel_patch(cor, p_est, scale=scale, roi_size=roi_size)), dims=dim)
        end
    end
    # return roi
    m,p = findmax(abs.(roi))
    roi_mid = (size(roi).÷2 .+ 1)
    return p_est .+ ((Tuple(p) .- roi_mid) ./ scale)
end


function get_subpixel_correl(dat;  other=nothing, k_est=nothing, psf=psf, upsample=true)
    up, up_other = prepare_correlation(dat; upsample=upsample, other=other, psf=psf)    
    optim_correl(up, up_other, k_est=k_est)
end


