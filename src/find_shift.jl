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

"""
FindWaveFit

Uses an gradient-based fitting based on fitting an exp(i k x) wave to the data.
"""
struct FindWaveFit <: FindMethod end

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

    # calculates the loss directly in Fourier-space using Parseval's theorem
    loss(v) = mynorm(fdat1, exp_shift_dat(fdat2, v)) # win .* 
    function g!(G, x)  # (G, x)
        G .= Zygote.gradient(loss, x)[1]
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

"""
    find_shift(dat1, dat2, Δx=nothing; zoom=nothing, mask1=nothing, mask2=nothing, est_pos=nothing, est_std=10.0, lambda=0.05, dims=1:ndims(dat1))

finds the integer shift with between two input images via the maximum of the cross-correlation.

Returned is an estimate of the real space (normally integer) pixel shift.

If the `zoom` argument (a number of a Tuple) is provided, in a second step, the inverse FFT is replaced by a chirp Z transformation
with the given zoom factor. This yields a result, which is by this factor more precise. A typical zoom factor
is for example 200, but it should not be bigger than the image size.
Note that for `zoom` being not nothing, the algorithm will also trim both arrays to their mutual overlap and apply a Hanning window over the whole field.

The first image is interpreted as the ground truth, whereas the second as a measurement.
Returned is the shift vector.

The optional argument Δx specifies a known integer shift, which then is used with the "zoom" argument to provide
-  a better sub-pixel estimate.

# Arguments
- `dat1`: first image
- `dat2`: second image
- `Δx`: known shift to find a more precise sub-pixel estimate for
- `zoom`: zoom factor for the sub-pixel estimate (if wanted)
- `mask1`: mask for dat1 to appyl when finding the shift
- `mask2`: mask for dat2 to appyl when finding the shift
- `est_pos`:  if provided, a Gaussian preference mask will be applied to bias towards a shift as defined by Δx
- `est_std`:  the standard deviation of the Gaussian preference mask (only used if `est_pos` was defined)
- `lambda`: a regularisation factor for the masked cross-correlation. It can be interpreted as approximately the minimum ratio of accepted overlap mask pixels.
- `exclude_zero`: if `true` the zero shift is excluded from the search
- `dims`: the dimensions to search for the shift in

# Example:
julia> img = rand(512,512);

julia> simg = shift(img, (3.3, 4.45));

julia> find_shift(simg, img)
(3, 4)

julia> find_shift(simg, img, zoom=100.0)
(3.3, 4.45)

julia> find_shift_iter(simg, img)
2-element Vector{Float64}:
 3.298879717301924
 4.449585689185147
"""
function find_shift(dat1, dat2, Δx=nothing; zoom=nothing, mask1=nothing, mask2=nothing, 
                                est_pos=nothing, est_std=10.0, lambda=0.05, exclude_zero=false, dims=1:ndims(dat1))
    if isnothing(Δx)
        mycor = let
            if (!isnothing(mask1) || !isnothing(mask2))
                if (isnothing(mask1))
                    mask1 = ones(eltype(dat1), size(dat1));
                else
                    dat1 = inpaint(dat1, mask1);  # no need to inpaint mask1 
                end
                if (isnothing(mask2))
                    mask2 = ones(eltype(dat2), size(dat2));
                else
                    dat2 = inpaint(dat2, mask2);   # no need to inpaint mask2 
                end
                cor_dat = get_correlation(dat1.*mask1, dat2.*mask2)
                cor_mask = get_correlation(mask1, mask2)
                abs2cormask = abs2.(cor_mask)
                cor_dat.*cor_mask./(abs2cormask .+ lambda*maximum(abs2cormask)) # a Tichonov-type mask to avoid division by zero
            else
                get_correlation(dat1, dat2)
            end
        end
        if (!isnothing(est_pos))
            preference_mask = gaussian_col(typeof(mycor), size(mycor); sigma=est_std, offset=(size(mycor).÷2 .+1).+est_pos)
            mycor .*= preference_mask;
        end
        Δx = find_max(mycor, exclude_zero=exclude_zero, dims=dims)
    end

    # find_peak(dat, fdat, Δx; method=:FindZoomFT, exclude_zero=true, max_range=nothing, scale=40, abs_first=false, roi_size=5, verbose=false, normalize=true)

    if !isnothing(zoom)
        if length(zoom) == 1
            zoom = ntuple(i -> zoom, Val(ndims(dat1)))
        end
        dat1c, dat2c = shift_cut(dat1, dat2, .- Δx)
        win = window_hanning(Float32, size(dat1c), border_in=0.0)
        fmul2 = ft(win .* dat1c) .* conj(ft(win .* dat2c))
        # this is quite slow. Would be nice to use a real-valued CZT where appropriate
        red_size = ceil.(Int, 2 .*zoom); # +- 0.5 pixel are allowed.
        mycor = iczt(fmul2, zoom, 1:ndims(fmul2), red_size)
        pos2 = find_max(abs2.(mycor), exclude_zero=false, dims=dims)
        return Δx .+ pos2 ./zoom
    end
    return Δx
end

"""
    get_correlation(dat1, dat2)

returns the real-valued correlation of dat1 with dat2, possibly expanding dat2 in size.
"""
function get_correlation(dat1, dat2)
    # calculate the forward part of the cross-correlation, up to the multiplication:
    fmul = let
        if all((size(dat2).-size(dat1)) .== 0)
            if eltype(dat1)<:Real && eltype(dat2)<:Real
                rfft(dat1) .* conj(rfft(dat2))
            else
                fft(dat1) .* conj(fft(dat2))
            end
        else 
            # make both images equal size
            dat2_big = select_region(dat2, new_size=size(dat1))
            # ref = select_region(ones(eltype(dat1), size(dat2)), new_size=size(dat1))
            if eltype(dat1)<:Real && eltype(dat2)<:Real
                rft1 = rfft(dat1)
                rft1 .* conj(rfft(dat2_big))
                # cor2 = fftshift(irfft(rft1 .* conj(rfft(ref)), size(dat1)[1]))
                # res = cor1 ./ (abs.(cor2) .+ maximum(cor1) ./ 100)
                # @vt cor1 res
            else
                rft1 = fft(dat1)
                rft1 .* conj(fft(dat2_big))
            end
        end
    end
    # find the maximum in the cross-correlation:
    mycor = let
        if eltype(dat1)<:Real && eltype(dat2)<:Real
            fftshift(irfft(fmul, size(dat1)[1]))
        else
            mycor = fftshift(ifft(fmul))
            abs2.(mycor)
        end
    end    
    return mycor
end


"""
    find_shift_iter(dat1, dat2, Δx=nothing)

finds the shift between two input images by minimizing the distance.
To be fast, this distance is calculated in Fourierspace using Parseval's theorem.
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
            find_shift(dat1, dat2) # to get a first estimate of the integer shift
        else
            Δx
        end
    end
    # @show Δx

    dat1c, dat2c = shift_cut(dat1, dat2, .-Δx)
    # return dat1, dat2, mycor, Δx
    win = window_hanning(Float32, size(dat1c), border_in=0.0)

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
"""
    find_ft_iter(dat, k_est=nothing; exclude_zero=true, max_range=nothing, verbose=false)

finds a peak in the Fourier transform of the input data `dat` by iteratively shifting and optimizing the peak strength.
In detail this is achieved by exploiting the Parseval theorem to calculate the sum of squared differences in Fourier space.
"""
function find_ft_iter(dat::AbstractArray{T,N}, k_est=nothing; exclude_zero=true, max_range=nothing, verbose=false, grad_tol=1e-7, maxiter=10, normalize=true) where{T,N}
    RT = real(T)
    win = collect(window_hanning(Float32, size(dat), border_in=0.0))
    wdat = win .* dat 

    k_est = let
        if isnothing(k_est)
            find_max(abs2.(ft(wdat)), exclude_zero=exclude_zero)
        else
            k_est
        end
    end
    k_est = RT.([k_est...])

    if (normalize)
        v_init = sqrt(abs2_ft_peak(wdat, k_est))
        if v_init != 0.0
            wdat ./= v_init
        else
            @warn "FT peak is zero."
        end
    end

    mynorm2(x) = -abs2_ft_peak(wdat, x)  # to normalize the data appropriately
    #@show mynorm(k_est)
    #@show typeof(k_est)
    #@show mygrad(k_est)
    function g!(G, x)  # (G, x)
        G .= Zygote.gradient(mynorm2, x)[1]
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
            optimize(od, lower, upper, k_est, Fminbox(), Optim.Options(show_trace=verbose, g_tol=grad_tol, iterations=maxiter)) # {GradientDescent}, , x_tol = 1e-2, g_tol=1e-3
        else
            optimize(od, k_est, LBFGS(), Optim.Options(show_trace=verbose, g_tol=grad_tol, iterations=maxiter)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
            # @time optimize(od, k_est, GradientDescent(), Optim.Options(show_trace=verbose, g_tol=1e-3, iterations=10)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
            # @time optimize(mynorm2, g!, k_est, LBFGS(), Optim.Options(show_trace=true)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
        end
    end
    # Optim.minimizer(res), mynorm(Optim.minimizer(res))
    # @show res
    k_res = res.minimizer
    peak_cpx = sum_exp_shift(dat, k_res)
    return k_res, angle(peak_cpx), abs.(peak_cpx) ./ length(dat)
end

"""
    get_sin_fit_fg!(dat)

returns an matching fg! function to be used in Optim for a sine function model.
Parameters:
+ dat: The data to construct the function for
+ init_scale: a factor to ensure that the initial step length is small enough.

"""
function get_sin_fit_fg!(dat; init_scale=10)
    N = length(dat)*init_scale
    ctr = size(dat) .÷ 2 .+ 1
    fscale = 2pi ./ size(dat)
    freq_ax = [reorient((axes(dat)[d] .- ctr[d]).* fscale[d], Val(d)) for d in 1:ndims(dat)]
    
    function fg!(F, G, vec)
        k = vec[1:end-2]
        Φ = vec[end-1]
        a = vec[end]
        # use the complex-valued exponential function, since it is separable. 
        # in the broadcasting operation the sine and cosine components are then separated.
        myexp = exp_ikx_sep(size(dat), shift_by=k)
        myexp.args[1] .*= exp.(1im * Φ) # a trick to incorporate the phase
        resid = a .* imag.(myexp) .- dat
        if G !== nothing
            # code to compute gradient here
            # grad k
            # coskxphi is real.(myexp)
            for d=1:length(k)
                myaxis = freq_ax[d]
                G[d] = .- 2*a*sum(resid .* real.(myexp) .* myaxis) / N
            end
            # grad Φ
            G[end-1] = 2*a*sum(resid .* real.(myexp)) / N
            # grad a
            G[end] = 2*sum(resid .* imag.(myexp)) / N
            # @show vec
            # @show G
        end
        if F !== nothing
            # value = ... code to compute objective function
            return sum(abs2.(resid)) / N
        end 
    end
    return fg!
end

# gradient-based fitting routine that fits a sine function to real-valued data
"""
    lsq_fit_sin(dat, k_init, phase_init=0f0, amp_init=1f0, exclude_zero=true)

fits a (potentially non-commensurate) sine function to N-dimensional data using Optim.jl
and a hard-coded model of the derivatives.

Parameters:
+ dat: input data to fit amp * sin(k x + phase) to.
+ k_init: a preexisting estimate of the shift_by argument of the sine function. This relates to k via
k = k_init .* 2pi ./ size(dat)
+ phase_init: the initial phase of the fit 
+ amp_init: the initial amplitude of the fit 
"""
function lsq_fit_sin(dat, k_init, phase_init=0f0, amp_init=1f0, exclude_zero=true)
    fg! = get_sin_fit_fg!(dat)

    init_vec = [(Float32.(k_init))..., phase_init, amp_init]
    # @show init_vec
    # @show fg!(1, nothing, init_vec)
    res = optimize(Optim.only_fg!(fg!), init_vec, LBFGS(), Optim.Options(iterations=10))
    print(res)
    res = Optim.minimizer(res)
    k_res = res[1:end-2]
    phi_res = res[end-1]
    a_res = res[end]
    return k_res, phi_res, a_res
end

"""
    get_exp_ikx_fit_fg!(dat; init_scale=10)

returns an matching fg! function to be used in Optim for a exp_ikx function model.
Parameters:
+ dat: The data to construct the function for
+ init_scale: a factor to ensure that the initial step length is small enough.
"""
function get_exp_ikx_fit_fg!(dat; init_scale=10)
    N = length(dat)*init_scale
    ctr = size(dat) .÷ 2 .+ 1
    fscale = 2pi ./ size(dat)
    freq_ax = [reorient((axes(dat)[d] .- ctr[d]).* fscale[d], Val(d)) for d in 1:ndims(dat)]
    
    function fg!(F, G, vec)
        k = vec[1:end-2]
        Φ = vec[end-1]
        a = vec[end]
        # use the complex-valued exponential function, since it is separable. 
        # in the broadcasting operation the sine and cosine components are then separated.
        myexp = exp_ikx_sep(size(dat), shift_by=k)
        myexp.args[1] .*= exp.(1im * Φ) # a trick to incorporate the phase
        resid = a .* myexp .- dat
        # print(" loss: $(sum(abs2.(resid)) / N), k: $(k)\n")
        if G !== nothing
            # code to compute gradient here
            # grad k
            # coskxphi is real.(myexp)
            for d=1:length(k)
                myaxis = freq_ax[d]
                G[d] = imag(.- 2*a*sum(resid .* conj.(myexp) .* myaxis) / N)
            end
            # grad Φ
            G[end-1] = imag(2*a*sum(resid .* conj.(myexp))) / N
            # grad a
            G[end] = real(2*sum(resid .* conj.(myexp))) / N
        end
        if F !== nothing
            # value = ... code to compute objective function
            return sum(abs2.(resid)) / N
        end 
    end
    return fg!
end

# gradient-based fitting routine that fits a sine function to real-valued data
"""
    lsq_fit_exp_ikx(dat, k_init, phase_init=0f0, amp_init=1f0; verbose=false, normalize=true)

fits a (potentially non-commensurate) exp_ikx function to N-dimensional data using Optim.jl
and a hard-coded model of the derivatives.

Parameters:
+ dat: input data to fit amp * sin(k x + phase) to.
+ k_init: a preexisting estimate of the shift_by argument of the sine function. This relates to k via
k = k_init .* 2pi ./ size(dat)
+ phase_init: the initial phase of the fit 
+ amp_init: the initial amplitude of the fit 
"""
function lsq_fit_exp_ikx(dat, k_init, phase_init=0f0, amp_init=1f0, maxiter=10; verbose=false, normalize=true)
    fg! = get_exp_ikx_fit_fg!(dat)

    normfac = 1
    if normalize
        normfac = mean(abs.(dat))
        dat ./= normfac
        amp_init /= normfac
    end
    init_vec = [(Float32.(k_init))..., phase_init, amp_init]
    # @show init_vec
    # @show fg!(1, nothing, init_vec)
    res = optimize(Optim.only_fg!(fg!), init_vec, LBFGS(), Optim.Options(iterations=maxiter, show_trace=verbose))
    # print(res)
    res = Optim.minimizer(res)
    k_res = res[1:end-2]
    phi_res = res[end-1]
    a_res = res[end] * normfac
    return k_res, phi_res, a_res
end

"""
    get_pixel_peak_pos(fdat; interactive=false, overwrite=true, exclude_zero=true)

returns the pixel position of the peak in the Fourier-transformed data `fdat`.
If `interactive` is `true`, the user is asked to click on the peak position.
If `overwrite` is `false`, the previously saved interactive click position will be used.
If `exclude_zero` is `true`, the zero pixel position in Fourier space is excluded, when looking for a global maximum.

returned is a tuple of the pixel position(s) of the peak (each a Tuple).
"""
function get_pixel_peak_pos(fdat; interactive=false, overwrite=true, exclude_zero=true)
    if interactive
        pos = get_positions(Float32.(abs.(fdat)), overwrite=overwrite)
        pos = Tuple((round.(Int, (p .- (size(fdat).÷ 2 .+1)))...,) for p in pos)
        # pos = find_max(abs2.(fdat), exclude_zero=exclude_zero)
        pos
    else
        find_max(abs2.(fdat), exclude_zero=exclude_zero)
        # RT.(find_max(abs2.(fdat), exclude_zero=exclude_zero))
    end

end

"""
    find_ft_peak(dat, k_est=nothing; method=:FindZoomFT::FindMethod, interactive=true, overwrite=true, exclude_zero=true, max_range=nothing, scale=40, abs_first=true, roi_size=10, ft_mask=nothing, verbose=false, normalize=true, reg_param=1e-6, show_quality=false, phase_only=false, reg_param_p=1e-8, psf=nothing)

localizes the peak in Fourier-space with sub-pixel accuracy using various routines, some involving fitting.
Returned is a tuple of vector of the peak position in Fourier-space as well as the phase and absolute amplitude of the peak.

# Arguments
+ `dat`:  data in real space, which will be Fourier transformed
+ `k_est`: a starting value for the peak-position. Needs to be accurate enough such that a gradient-based search can be started with this value
        If `nothing` is provided, an FFT will be performed and a maximum is chosen. 
+ `method`: defines which method to use for finding the maximum. Current options:
    `:FindZoomFT`:  Uses a chirp Z transformation to zoom into the peak. 
    `:FindIter`:    Uses an iterative optimization via FFT-shift operations to find the peak.
    `:FindWaveFit`: Uses a gradient-based fitting of a complex exponential to the input data to find the peak.
    `:FindShiftedWindow`: Uses a method that shifts a real-space window and sums over it to find the peak in Fourier-space.
                Note that there can be issues with a multi-peaked modification in real space. But it is quite fast.
    `:FindCOM`: Uses an FFT and uses a 3x3x... region to localize its absolute value via a center of mass method (subtracting the minimum in this region).
                Note that this does not work well for very narrow peaks.
    `:FindParabola`: Uses an FFT and uses a 3x3x... region to localize its absolute value via fitting a parbola through the integer center along each dimension.
                Note that this does not work well for very narrow peaks.
+ `interactive`: a boolean defining whether to use user-interaction to define the approximate peak position. Only used if `k_est` is `nothing`.
+ `ignore_zero`: if `true`, the zero position will be ignored. Only used if `!isnothing(k_est)`.
+ `max_range`: maximal search range for the iterative optimization to be used as a box-constraint for the `Optim` package.
+ `overwrite`: if `false` the previously saved interactive click position will be used. If `true` the previous value is irnored and the user is asked to click again.
+ `max_range`: maximal search range to look for in the +/- direction. Default: `nothing`, which does not restrict the local search range.
+ `exclude_zero`: if `true`, the zero pixel position in Fourier space is excluded, when looking for a global maximum
+ `scale`: the zoom scale to use for the zoomed FFT, if `method==:FindZoomFT`
+ `ft_mask`: a mask to multiply to the Fourier-transformed function. This can be used to mask out certain regions in the correlation function.
+ `normalize`: if `true` the data will be normalized for the iterative methods to prevent to large gradients 
+ `roi_size`: the size of the region of interest to use for the iterative methods. This is only used if `method` is `:FindIter` or `:FindShiftedWindow`.
+ `verbose`: if `true`, the optimization will print out information about the optimization process.
+ `show_quality`: if `true`, the peak contrast will be printed out, which is the ratio of the peak height to the base height surrounding the peak.
+ `phase_only`: if `true`, the product in Fourier space as represented by `dat` is first normalized to an absolute of one. This yields a phase-only correlation.
+ `reg_param`: a regularization parameter to avoid division by zero in the Fourier space. This is only used if `psf` is provided.
+ `reg_param_p`: a regularization parameter to avoid division by zero in the Fourier space, if `phase_only` is true. Default: 1e-8.
"""
function find_ft_peak(dat::AbstractArray{T,N}, k_est=nothing; method=:FindZoomFT, psf=nothing, interactive=false, overwrite=true, exclude_zero=true, max_range=nothing, scale=40, abs_first=false, roi_size=5, verbose=false, ft_mask=nothing, reg_param=1e-6, reg_param_p=1e-8, normalize=true, show_quality=false, phase_only=false) where{T,N}
    # @show k_est
    # fdat = nothing
    # RT = real(T)
    #if isnothing(fdat)
    if !isnothing(psf)
        fourier_weigths = ft(abs2.(psf))
        fourier_weigths = conj.(fourier_weigths)./(abs2.(fourier_weigths) .+ reg_param)
        if !isnothing(ft_mask)
            ft_mask .= fourier_weigths .* correl_mask
        end
    end
    if phase_only
        dat = dat ./ (abs.(dat) .+ reg_param_p) # normalize to absolute of one
        # p_only(x) = ifelse(x == 0, x, x / abs(x))  # makes no difference
        # dat = p_only.(dat)
    end
    fdat = ft(dat)
    if !isnothing(ft_mask)
        fdat = fdat .* ft_mask
        # @show size(fdat)
    end

    #end
    if isnothing(k_est)
        k_est = get_pixel_peak_pos(fdat, interactive=interactive, overwrite=overwrite, exclude_zero=exclude_zero)
    end

    if (show_quality)
        k_pos = round.(Int, k_est[1]) .+ (size(fdat) .÷ 2 .+ 1)
        delta_k = 6 .* ones(length(k_pos))
        peak_height = abs.(fdat[k_pos...])
        base_height = 0.0
        for d=1:length(k_pos)
            k_plus = [k_pos...]; k_plus[d] += delta_k[d]
            k_minus = [k_pos...]; k_minus[d] -= delta_k[d]
            base_height += abs.(fdat[k_plus...]) .+ abs.(fdat[k_minus...])
        end
        base_height /= 2 .* length(k_pos) # average over the number of dimensions
        peakcontrast = (peak_height .- base_height) ./ base_height;
        println("Peak contrast @$k_pos = $peak_height is $peakcontrast")
        # println("Peak $peak_height")
        # println("Base $base_height")
    end

    if isa(k_est[1], AbstractArray) || isa(k_est[1], Tuple) # Array of positions
        res_k = []
        res_phase = []
        res_int = []
        for k in k_est
            res = find_peak(dat, fdat, k; method=method, exclude_zero=exclude_zero, max_range=max_range, scale=scale, abs_first=abs_first, roi_size=roi_size, verbose=verbose, normalize=normalize);
            push!(res_k, res[1])
            push!(res_phase, res[2])
            push!(res_int, res[3])
        end
        return res_k, res_phase, res_int
    else
        return find_peak(dat, fdat, k_est; method=method, exclude_zero=exclude_zero, max_range=max_range, scale=scale, abs_first=abs_first, roi_size=roi_size, verbose=verbose, normalize=normalize)
    end

    # @show k_est
    # @show k_est
    # k_est = [k_est...]
end

function get_com(fdat, k_est) # should one use abs or abs2?
    k_est = Tuple(round.(Int, k_est))
    ctr = size(fdat) .÷ 2 .+ 1
    ROI = (abs.(select_region_view(collect(fdat), ntuple((d)->3, ndims(fdat)); center=k_est.+ ctr))) # .^0.2
    ROI = ROI .- minimum(ROI)
    t = ntuple((d)->0, ndims(fdat))
    for ci in CartesianIndices(size(ROI))
        t = t .+ Tuple(ci) .* ROI[ci]
    end
    return (t./sum(ROI)) .- 2 .+ k_est
end

function get_parabola_max(fdat, k_est) # should one use abs or abs2?
    k_est = Tuple(round.(Int, k_est))
    ctr = size(fdat) .÷ 2 .+ 1
    ROI = (abs.(select_region_view(collect(fdat), ntuple((d)->3, ndims(fdat)); center=k_est.+ ctr))) #.^0.2
    res = zeros(length(k_est))
    for d=1:ndims(fdat)
        y1 = ROI[ntuple((i)->(i==d) ? 1 : 2, ndims(fdat))...]
        y2 = ROI[ntuple((i)->(i==d) ? 2 : 2, ndims(fdat))...]
        y3 = ROI[ntuple((i)->(i==d) ? 3 : 2, ndims(fdat))...]
        a = (y1 + y3 - 2*y2)/2
        b = (y3 -y1)/2
        # c = y2
        res[d] = -b / a # (y3 - y1)/(y1+y3+2*y2)
        # mymax = a*(res[d])^2+b*res[d]+c
    end
    return Tuple(res) .+ k_est
end

"""
    find_peak(dat, fdat, k_est; method=:FindZoomFT, exclude_zero=true, max_range=nothing, scale=40, abs_first=false, roi_size=5, verbose=false)

    finds a peak in Fourier space using various strategies. Some need FFTs and others not.
+ method: defines which method to use for finding the maximum. Current options:
    `:FindZoomFT`:  Uses a chirp Z transformation to zoom into the peak. 
    `:FindIter`:    Uses an iterative optimization via FFT-shift operations to find the peak.
    `:FindWaveFit`: Uses a gradient-based fitting of a complex exponential to the input data to find the peak.
    `:FindShiftedWindow`: Uses a method that shifts a real-space window and sums over it to find the peak in Fourier-space.
                Note that there can be issues with a multi-peaked modification in real space. But it is quite fast.
    `:FindCOM`: Uses an FFT and uses a 3x3x... region to localize its absolute value via a center of mass method (subtracting the minimum in this region).
                Note that this does not work well for very narrow peaks.
    `:FindParabola`: Uses an FFT and uses a 3x3x... region to localize its absolute value via fitting a parbola through the integer center along each dimension.
                Note that this does not work well for very narrow peaks.
    `:FindShiftedWindow`: Uses a shifted window of the non-fted data to find the peak.

+ dat:  data in real space, which will be Fourier transformed
+ fdat: the Fourier-transformed data. If `nothing` is provided, the Fourier transform will be calculated (if needed).
+ k_est: a starting value for the peak-position. Needs to be accurate enough such that a gradient-based search can be started with this value
        If `nothing` is provided, an FFT will be performed and a maximum is chosen.
+ exclude_zero: if `true`, the zero pixel position in Fourier space is excluded, when looking for a global maximum
+ max_range: maximal search range to look for in the +/- direction. Default: `nothing`, which does not restrict the local search range.
+ scale: the zoom scale to use for the zoomed FFT, if `method==:FindZoomFT`
+ abs_first: if `true`, the absolute is calulated first.
+ roi_size: the size of the region of interest to use for the center of mass.
+ verbose: if `true`, the optimization routine will print out the progress.
+ normalize: if `true` the data will be normalized for the iterative methods to prevent to large gradients
"""
function find_peak(dat, fdat, k_est; method=:FindZoomFT, exclude_zero=true, max_range=nothing, scale=40, abs_first=false, roi_size=5, verbose=false, normalize=true)
    # @show k_est
    if method == :FindCOM # center of mass based determination of the maximum
        if isnothing(fdat)
            fdat = ft(dat)
        end
        kg = get_com(fdat, k_est)
        peak_cpx = sum_exp_shift(dat, kg)
        return kg, angle(peak_cpx), (abs.(peak_cpx) ./ length(dat))
    elseif method == :FindParabola # parabola fit to the peak
        if isnothing(fdat)
            fdat = ft(dat)
        end
        kg = get_parabola_max(fdat, k_est)
        peak_cpx = sum_exp_shift(dat, kg)
        return kg, angle(peak_cpx), (abs.(peak_cpx) ./ length(dat))
    elseif method == :FindZoomFT # use CZT to zoom in and pick the maximum in the zoomed region
        # k_est = Tuple(round.(Int, k_est))
        if isnothing(fdat)
            fdat = ft(dat)
        end
        k_est = Tuple(round.(Int, k_est))
        kg, _, _ = get_subpixel_peak(fdat, k_est, scale=scale, exclude_zero=exclude_zero, abs_first=abs_first, roi_size=roi_size)
        peak_cpx = sum_exp_shift(dat, kg) # re-calculate angle and 
        return kg, angle(peak_cpx), (abs.(peak_cpx) ./ length(dat))
    elseif method == :FindIter # use iterative optimization to find the peak
        k_est = Tuple(round.(Int, k_est))
        return find_ft_iter(dat, k_est; exclude_zero=exclude_zero, max_range=max_range, verbose=verbose, normalize=normalize);
    elseif method == :FindWaveFit # does not need fdat. Uses the data directly via iterative optimization
        k_est = Tuple(round.(Int, k_est))
        k_pos = round.(Int, k_est) .+ size(fdat) .÷ 2 .+ 1
        if isnothing(fdat)
            fdat = ft(dat)
        end
        val = fdat[k_pos...]
        phase_init = angle(val)
        amp_init = abs(val) / length(dat)
        res = lsq_fit_exp_ikx(dat, .-k_est, phase_init, amp_init, verbose=verbose, normalize=normalize);
        # @show typeof(fdat)
        # @show typeof(res)
        res[1] .= .- res[1]
        return res # return the full information
    elseif method == :FindShiftedWindow # uses a shifted window of the non-fted data to find the peak.
        return subpixel_kg(dat, k_est)
    else
        error("Unknown method for localizing peak. Use :FindZoomFT or :FindIter")
    end
end

"""
    subpixel_kg(data, k0)

Estimate the subpixel position of a peak in a multidimensional array.
Parameters:
    data: array of the data
    k0: integer Fourierspace position of the peak
"""
function subpixel_kg(data, k0)
    sz = size(data)
    # generate an object that looks like an array representing an exponential shifting operation
    # Yet the separability of such an exponential is exploitet to make to code faster.
    my_nd_exp = exp_ikx_sep(sz, shift_by=k0)
    kg = zeros(length(k0))
    shift_one_pixel = exp_ikx_sep(sz, shift_by=ones(length(k0)))
    win_shift = 1
    for dim=1:ndims(data)
        if sz[dim] == 1
            kg[dim] = k0[dim]
            continue
        end
        # list all projection dimensions
        alldims = ntuple(d -> (d<dim) ? d : d+1, ndims(data)-1)
        # project over all dimensions except the current one, but also shifts the peak back to the center with integer accuracy
        proj = sum(my_nd_exp .* data, dims=alldims)
        # create a one-dimensional Hanning window over N-1 points
        win = (1 .+ cos.(range(-pi, stop=pi, length=sz[dim]-win_shift))) ./ 2
        # win = window_hanning((sz[dim],).-1) # needs a tuple as size input
        sum1 = sum(proj[1:end-win_shift] .* win)
        sum2 = sum(proj[1+win_shift:end] .* win)
        rel_k = (sz[dim]-win_shift)*(angle(sum2)-angle(sum1))/(2pi)/win_shift
        sum1 = sum(proj[1:end-win_shift] .* shift_one_pixel.args[dim][1:end-win_shift] .* win)
        sum2 = sum(proj[1+win_shift:end] .* shift_one_pixel.args[dim][1+win_shift:end] .* win)
        rel_km1 = (sz[dim]-win_shift)*(angle(sum2)-angle(sum1))/(2pi)/win_shift
        sum1 = sum(proj[1:end-win_shift] .* conj.(shift_one_pixel.args[dim][1:end-win_shift]) .* win)
        sum2 = sum(proj[1+win_shift:end] .* conj.(shift_one_pixel.args[dim][1+win_shift:end]) .* win)
        rel_kp1 = (sz[dim]-win_shift)*(angle(sum2)-angle(sum1))/(2pi)/win_shift
        # @show dim
        # @show sz[dim]
        if abs((rel_kp1 - rel_km1) - 2.0) > 0.1
            # @warn "Problem in phase-based subpixel estimation using the :FindShiftedWindow method. Peak too broad. Using integer maximum instead."
            # works for some peaks but not for all
            kg[dim] = k0[dim] + rel_k / ((rel_kp1 - rel_km1)/2)    #*sz[1]/2π
            # kg[dim] = k0[dim]
        else
            kg[dim] = k0[dim] + rel_k # / ((rel_kp1 - rel_km1)/2)    #*sz[1]/2π
        end
        # @show kg[dim] = k0[dim] + (rel_kp1+rel_km1)/win_shift/(rel_kp1 - rel_km1)    #*sz[1]/2π
    end
    # @show kg
    peak_cpx = sum_exp_shift(data, kg)
    return kg, angle(peak_cpx), (abs.(peak_cpx) ./ length(data))
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

juliat> myerr = [shifts[d][1:2] .- sh[d] for d in 1:length(sh)]
```
"""
function align_stack(dat::AbstractArray{T,N}; refno=nothing, ref = nothing, damp=0.1, max_freq=0.4, dim=ndims(dat), method=:FindIter, shifts=nothing) where{T,N}
    RT = real(T)
    refno = let
        if isnothing(refno)
            size(dat)[dim] ÷2 +1
        else
            refno
        end
    end
    if eltype(dat) <: Integer
        dat = Float32.(dat)
    end
	ref = let
		if isnothing(ref)
			slice(dat, dim, refno)
		else
			ref
		end
	end
	# damp_edge_outside(ref, damp)
    # do not use RT below, since then these algorithms do not work with Int as input type
	wh = window_hanning(Float32, size(ref), border_in=0.0, border_out=1.0-damp)
	fwin = window_radial_hanning(Float32, size(ref), 		 
  		border_in=max_freq*0.8,border_out=max_freq*1.2)
		# rr(size(ref)) .< maxfreq .* size(ref)[1]
	# imgs = []
	res_shifts = []
	fref = fwin .* conj(ft(wh .* (ref .- mean(ref))))
    res = similar(dat, RT)
	for n=1:size(dat,dim)
        aslice = slice(dat, dim, n)
        res_slice = slice(res, dim, n)
        # aslice = dropdims(aslice, dims=dim)
        myshift = let 
            if isnothing(shifts)
                cor_ft = ft(wh .* (aslice .- mean(aslice))) .* fref
                sh, pha, am = find_ft_peak(cor_ft, method=method, interactive=false, exclude_zero=false)
                sh
            else
                shifts[n]
            end
        end
		# m, p = findmax(abs.(cor))
		# midp = size(cor).÷ 2 .+1
		# myshift = midp .- Tuple(p)
		push!(res_shifts, myshift)
        res_slice .=  shift(aslice, myshift)
		# push!(imgs, shift(aslice, myshift))
		# push!(imgs, cor_ft)
	end
	# cat(imgs...;dims=dim), res_shifts
	res, res_shifts
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

# """
#     optim_correl(dat, other=dat; k_est = nothing, method=:FindZoomFT, verbose=false, correl_mask=nothing)

# optimizes the correlation of Fourier-transformed `dat` with `other` by maximizing the correlation strength.
# Note that you should probably be using "get_subpixel_correl" instead, which accepts real-space data and preprocesses it.

# # Arguments
# + dat: the data already in Fourier space to correlate (using one additional Fourier transformation)
# + other: the reference data to correlate with, as a default the data is autocorrelated.
# + k_est: an initial estimate of the shift
# + method: the method to use for finding the peak
# + verbose: if `true` the optimization process is printed
# + correl_mask: a mask to apply to the correlation to restrict the search space

# Returned is the optimal shift and the correlation strength and phase at this shift.

# # Example
# ```julia
# julia> dat = rand(100,100);

# julia> simg = shift(dat, (3.3, 4.45));

# julia> optim_correl(simg, dat)
# ```
# """
# function optim_correl(dat, other=dat; k_est = nothing, interactive=false, method=:FindZoomFT, verbose=false, correl_mask=nothing)
#     if true
#         find_ft_peak(dat .* conj.(other), k_est, method=method, verbose=verbose, ft_mask=correl_mask, interactive=interactive)
#     else # old version below
#         mynorm(x) = - all_correl_strengths(x, dat, other)
#         #@show k_est
#         lower = k_est .- 2.1
#         upper = k_est .+ 2.1
#         mygrad(x) = Zygote.gradient(mynorm,x)[1]
#         function g!(G, x)
#             G .= mygrad(x)
#         end
#         od = OnceDifferentiable(mynorm, g!, k_est)
#         res = optimize(od,lower, upper, k_est, Fminbox()) # {GradientDescent}
#         # res = optimize(mynorm, k_est, LBFGS()) #NelderMead()
#         Optim.minimizer(res)
#     end
# end

"""
     prepare_correlation(dat; upsample=true, other=nothing, psf=nothing)

prepares the correlation by optionally convolving `dat` with the `psf` and upsampling the result by a factor of two.
This is needed since the correlation can extend twice as far.
`dat` should already be the inverse Fourier transformation of the data to correlate.
To correlate the Fourier transforms of real-valued data, just supply the real-space data as `dat`.
"""
function prepare_correlation(dat, psf=nothing; upsample=true)
    dat = let
        if !isnothing(psf)
            conv_psf(dat, psf)
        else
            dat
        end
    end
    up = let
        if upsample
                up = zeros(size(dat,1)*2, size(dat,2)*2, size(dat,3))
                for n=1:size(dat,3)
                    up[:,:,n] .= upsample2(dat[:,:,n])
                end
                up
        else
            dat
        end
    end

    return up
end

# note that dat and other are already in Fourier-space
"""
    correlate(dat; upsample=true, other=nothing, phase_only=false, psf=nothing)

correlates `dat` with a reference `other` or by default with itself.
`dat` should already be the inverse Fourier transformation of the array to correlate.
To correlate the Fourier transforms of real-valued data, just supply the real-space data as `dat`.
"""
function correlate(dat; upsample=true, other=nothing, phase_only=false, psf=nothing)
    up = prepare_correlation(dat, psf; upsample=upsample)
    up_other = let 
        if !isnothing(other)
            prepare_correlation(other, psf; upsample=upsample)
        else
            up  # autocorrelate
        end
    end

    ftcorrel = up .* conj.(up_other)
    # ctrpos = size(ftcorrel) .÷ 2 .+ 1
    # ftcorrel[ctrpos...] = zero(eltype(ftcorrel)) # ensures that both arrays are zero mean for correlation

    if phase_only
        ftcorrel ./= ifelse.(iszero.(ftcorrel), eltype(ftcorrel)(Inf), (abs.(ftcorrel)))
    end
    mycor = ft2d(ftcorrel)  # correlates starting in Fourier space
    # mycor = ft2d(abs2.(up))  # correlates starting in Fourier space
    return mycor
end

#  conclusion: cc seems to be the most reliable correlation

# now we try to find the subpixel position using a chirped z-transform
"""
    get_subpixel_patch(cor::AbstractArray{T,N}, p_est; scale=10, roi_size=5) where{T,N}

finds a subpixel patch of the correlation `cor` at the position `p_est` using a chirped z-transform (CZT).
To this aim, the input correlation is first centered (and potentially cropped) and then Fourier transformed 
and finally inverse Fourier transformed with a chirped z-transform, which zooms in by the factor `scale`.

# Arguments
+ cor: the correlation to zoom into
+ p_est: the position to zoom into
+ scale: the zoom factor to use (default: 10)
+ roi_size: the size of the region of interest to extract during this procedure
"""
function get_subpixel_patch(cor::AbstractArray{T,N}, p_est; scale=10, roi_size=5) where{T,N}
    if !(any(isinteger.(scale)))
        error("Scale needs to be an integer.")
    end
    scale = Tuple(scale .* ones(Int, ndims(cor)))
    p_mid = size(cor).÷2 .+ 1

    # determine the size to extract from cor
    # new_size = ceil.(Int, roi_size .* scale) # , size(cor))
    new_size = ntuple(d -> ifelse(size(cor,d) == 1, 1, ceil.(Int, roi_size .* scale[d])),  Val(ndims(cor)))

    # 2 .* scale .* ones(Int, ndims(cor)))))
    # if length(roi_size) < 2
    #     @show new_size = Tuple(roi_size .* scale for n in 1:ndims(cor))
    # else
    #     @show new_size = roi_size .* scale
    # end

    roi = select_region(cor, center=p_mid.+p_est, new_size=new_size)  # (iczt(fc .* exp_ikx(size(cor)[1:2], shift_by=.-p_est), scale)
    # @show scale = size(cor) ./ new_size
    # return roi
    fc = ft2d(roi .* window_hanning(Float32, size(roi), border_in=0.0))
    # fc = ft2d(roi)
    # fc = expand_dims(fc, Val(3))
    scale_mod = Tuple(ifelse((d<3), Float32(scale[d]), 1f0) for d in 1:length(scale))
    zoomed = iczt(fc, scale_mod, (1, 2))
    return zoomed
end

"""
    get_subpixel_peak(cor::AbstractArray{T,N}, p_est=nothing; exclude_zero=true, scale=10, roi_size=4, abs_first=false, dim=3) where{T,N}

obtains the subpixel peak position of the correlation `cor` with an initial estimate `p_est` using a chirped z-transform (CZT).
"""
function get_subpixel_peak(cor::AbstractArray{T,N}, p_est=nothing; exclude_zero=true, scale=10, roi_size=4, abs_first=false, dim=3) where{T,N}
    p_est = let
        if isnothing(p_est)
            find_max(abs2.(cor), exclude_zero=exclude_zero)
        else
            p_est
        end
    end
    # @show roi_size
    roi = let
        if abs_first 
            get_subpixel_patch(sum(abs.(cor), dims=dim), p_est, scale=scale, roi_size=roi_size)
        else
            get_subpixel_patch(cor, p_est, scale=scale, roi_size=roi_size)
                # else
            #     sum(abs.(get_subpixel_patch(cor, p_est, scale=scale, roi_size=roi_size)), dims=dim)
            # end
        end
    end
    # @show abs.(roi)
    # return roi
    m,p = findmax(abs2.(roi))
    peak_val = roi[p]
    roi_mid = (size(roi).÷2 .+ 1)
    return ((p_est .+ ((Tuple(p) .- roi_mid) ./ scale))...,), angle(peak_val), abs(peak_val) ./ length(cor)
end

"""
    get_rel_subpixel_correl(ft_order0, ft_shifted_order, kshift::NTuple, fwd_psf; fwd_psf2 = conv_psf(fwd_psf, fwd_psf), upsample=true)

returns the complex correlation coefficient of `shifted_order` normalized in relation to the central (unshifted) order0.
The orders `ft_order0` and `ft_shifted_order`, both provided as Fourier transforms of the data to correlate, are assumed to have been modified by a point spread function `fwd_psf` and shifted by `kshift`.
In a first step, they are pre-filtered to optimize the SNR in their correlation.
Note that the shift vector `k_shift` needs to be provided with subpixel accuracy.
"""
function get_rel_subpixel_correl(ft_order0, ft_shifted_order, kshift::NTuple, fwd_psf; fwd_psf2 = isnothing(fwd_psf) ? nothing : conv_psf(fwd_psf, fwd_psf), upsample=true)
    myexp = exp_ikx_sep(size(ft_order0), shift_by=kshift)
    if isnothing(fwd_psf2)
        fwd_psf2 = collect(myexp) 
    else
        fwd_psf2 .*= myexp 
    end
    up_shifted_order = prepare_correlation(ft_shifted_order, fwd_psf; upsample=upsample)
    up_order0 = prepare_correlation(ft_order0, fwd_psf; upsample=upsample)

    sum_shifted_order_exp = sum(up_shifted_order .* conj.(up_order0) .* myexp)
    # factor = correl / sum(wf * otf * conj(wf) * otf_shifted). Since wf already contains an OTF, the product cancels out.
    # We only need to account for the fact that the OTF is shifted via the other_psf_argument. This shifting is performed via k_other_psf_shift.
    up_wf_noconf  = prepare_correlation(ft_order0, nothing; upsample=upsample)
    up_order0_confshifted = prepare_correlation(ft_order0, fwd_psf2; upsample=upsample)
    sum_order0_exp = sum(up_wf_noconf .* conj(up_order0_confshifted))
    return sum_shifted_order_exp / sum_order0_exp
end

function get_rel_subpixel_correl(dat, other, shift_vecs::Vector, psf; upsample=true)
    res = []
    for k in shift_vecs
        push!(res, get_rel_subpixel_correl(dat, other, k, psf; upsample=upsample))
    end
    return res
end

"""

    get_subpixel_correl(dat;  other=nothing, k_est=nothing, psf=nothing, upsample=true, interactive=false, correl_mask=nothing, method=:FindZoomFT, verbose=false, reg_param=1e-6, show_quality=false)

returns the subpixel correlation of `dat` with `other` or by default with itself.

# Arguments
+ `dat`: the data to correlate with itself or `other` in (inverse) Fourier space. This is the real space data when correlating FT orders. 
+ `other`: the reference data to correlate with. If `nothing` the data is correlated with itself.
+ `k_est`: an initial estimate of the shift. If `nothing` the shift is determined via a cross-correlation.
+ `k_other_psf_shift`: if not nothing, the other data is filtered by a shifted OTF.
+ `psf`: the point spread function to use for prefiltering the data to correlate. If `nothing` no prefiltering is performed.
    Prefiltering can be useful to reduce noise and also uncorrelated background (if `psf` suppresses low frequencies).
+ `upsample`: if `true` the data is upsampled by a factor of two before correlation. This is necessary to avoid wrap-around effects.
+ `interactive`: if `true`, the user is asked to click on the approximate peak position in the correlation. Only used if `k_est` is `nothing`.
+ `correl_mask`: a mask to multiply to the Fourier-transformed function. This can be used to mask out certain regions in the correlation function.
+ `method`: the method to use for finding the peak in the correlation. Default: `:FindZoomFT`. Other options are:
    - `:FindIter`: uses an iterative optimization to find the peak.
    - `:FindWaveFit`: uses a gradient-based fitting of a complex exponential to the input data to find the peak.
    - `:FindShiftedWindow`: uses a shifted window of the non-fted data to find the peak.
    - `:FindCOM`: uses a center of mass method to find the peak.
    - `:FindParabola`: fits a parabola through the integer center along each dimension.
+ `verbose`: if `true`, the optimization will print out information about the optimization process.
+ `reg_param`: a regularization parameter to avoid division by zero in the Fourier space.
+ `show_quality`: if `true`, the peak contrast will be printed out, which is the ratio of the peak height to the base height surrounding the peak.
+ `phase_only`: limits the correlation to the phase only, i.e. the product in Fourier space as represented by `dat` is first normalized to an absolute of one.
    This yields a phase-only correlation.
+ `reg_param_p`: a regularization parameter to avoid division by zero in the Fourier space, if `phase_only` is true. Default: 1e-8.

returns the subpixel position of the correlation peak and the phase and amplitude of the peak.
"""
function get_subpixel_correl(dat;  other=nothing, k_est=nothing, psf=nothing, upsample=true, interactive=false, correl_mask=nothing, method=:FindZoomFT, verbose=false, reg_param=1e-6, show_quality=false, phase_only=false, reg_param_p=1e-8)
    if isnothing(other)
        other = dat
    end
    up = prepare_correlation(dat, psf; upsample=upsample)
    up_other = prepare_correlation(other, psf; upsample=upsample)

    ftcorrel = up .* conj.(up_other)
    res = find_ft_peak(ftcorrel, k_est; method=method, psf=psf, verbose=verbose, ft_mask=correl_mask, interactive=interactive, reg_param=reg_param, show_quality=show_quality, phase_only=phase_only, reg_param_p=reg_param_p)

    return res
end


