module FindShift
export abs2_ft_peak, sum_exp_shift, find_ft_peak, correlate, beautify, get_subpixel_peak, align_stack

using FourierTools, IndexFunArrays, NDTools, Optim, Zygote, LinearAlgebra, ChainRulesCore, Statistics
using View5D # for get_positions

# add FourierTools,IndexFunArrays, NDTools, Optim, Zygote, LinearAlgebra, ChainRulesCore
# add View5D, TestImages, Noise, FiniteDifferences

# since Zygote cannot deal with exp_ikx from the IndexFunArray Toolbox, here is an alternative
function exp_shift(sz, k_0)
    mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    [exp((1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(sz)]
end

function sum_exp_shift(dat, k_0)
    sz = size(dat)
    mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    sum(dat[p] * exp((1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat))
end

function sum_exp_shift_ix(dat, k_0)
    sz = size(dat)
    mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    accumulate(.+, dat[p] .* Tuple(p) .* exp((1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat))
end

"""
    abs2_ft_peak(dat, k_cur, dims=(1,2))
estimates the complex value of a sub-pixel position defined by `k_cur` in the Fourier-transform of `dat`.
"""
function abs2_ft_peak(dat, k_cur)
    abs2(sum_exp_shift(dat, Tuple(k_cur)))
end

 # define custom adjoint for sum_exp_shift
 function ChainRulesCore.rrule(::typeof(abs2_ft_peak), dat, k_cur)
    Y = sum_exp_shift(dat, k_cur)

    abs2_ft_peak_pullback = let sz = size(dat)
        function abs2_ft_peak_pullback(barx)
            # @show k_cur
            # @show abs2(Y)
            sum_dr_cos_di_sin = sum_di_cos_dr_sin  = zero(eltype(k_cur)); 
            sum_xdr_sin_xdi_cos = sum_xdi_sin_xdr_cos = zeros(eltype(k_cur), length(k_cur))
            pvec = 2pi .*Vector(k_cur) ./ sz;
            mymid = (sz.÷2).+1
            sp = coskx = sinkx = dr = di = zero(eltype(k_cur))        
            for p in CartesianIndices(dat)
                x = Tuple(p) .- mymid
                sp = dot(pvec, x)
                coskx = cos(sp); 
                sinkx = sin(sp);
                dr = real(dat[p]); 
                di = imag(dat[p]);
                dr_cos_di_sin = dr*coskx - di*sinkx
                di_cos_dr_sin = di*coskx + dr*sinkx
                sum_dr_cos_di_sin += dr_cos_di_sin
                sum_di_cos_dr_sin += di_cos_dr_sin
                sum_xdr_sin_xdi_cos .-= (2pi .* x./sz) .* di_cos_dr_sin # 
                sum_xdi_sin_xdr_cos .+= (2pi .* x./sz) .* dr_cos_di_sin # (x./sz)
            end
            res = 2 .*(sum_dr_cos_di_sin.*sum_xdr_sin_xdi_cos .+ sum_di_cos_dr_sin.*sum_xdi_sin_xdr_cos)
            # @show barx .* res
            #@show Vector(barx .* res)
            #@show barx
            #@show k_cur
            # return zero(eltype(dat)), zeros(eltype(dat), size(dat)) , Vector(barx .* res)
            return NoTangent(), (ChainRulesCore.@not_implemented "Save computation"), barx .* res
        end
    end
   # @show abs2(Y)
    return abs2(Y), abs2_ft_peak_pullback
end

function find_max(dat; exclude_zero=true)
    arr = abs.(dat)
    mid = center(size(arr), CenterFT)
    if exclude_zero
        arr[mid...] = 0
    end
    m,p = findmax(arr)
    Tuple(p) .- mid
end


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


function find_ft_iter(dat, k_est; exclude_zero=true, max_range=2.1)
    win = 1.0 # collect(window_hanning(size(dat), border_in=0.0))
    v_init = sqrt(abs2_ft_peak(win .* dat, k_est))
    mynorm(x) = -abs2_ft_peak(win .* dat ./ v_init, x)  # to normalize the data appropriately
    #@show mynorm(k_est)
    #@show typeof(k_est)
    #@show mygrad(k_est)
    function g!(G, x)  # (G, x)
        G .= gradient(mynorm,x)[1]
    end
    od = OnceDifferentiable(mynorm, g!, k_est)
    #@show fieldnames(typeof(od))
    res = let
        if false
            lower = k_est .- max_range
            upper = k_est .+ max_range
            @time optimize(od,lower, upper, k_est, Fminbox(), Optim.Options(show_trace=false, x_tol = 1e-2)) # {GradientDescent}, , g_tol=1e-3
        else
            @time optimize(od, k_est, LBFGS(), Optim.Options(show_trace=false, x_tol = 1e-2)) #NelderMead(), iterations=2, g_tol=1e-3
        end
    end
    # Optim.minimizer(res), mynorm(Optim.minimizer(res))
    res.minimizer
end



"""
    find_ft_peak(dat, k_est=nothing, ignore_zero=true, dims=(1,2))
    localizes the peak in Fourier-space with sub-pixel accuracy using an iterative fitting routine.

# Arguments
+ dat:  data
+ k_est: a starting value for the peak-position. Needs to be accurate enough such that a gradient-based search can be started with this value
        If `nothing` is provided, an FFT will be performed and a maximum is chosen. 
+ ignore_zero: if `true`, the zero position will be ignored. Only used if `!isnothing(k_est)`.
+ max_range: maximal search range for the iterative optimization to be used as a box-constraint for the `Optim` package.
+ overwrite: if `false` the previously saved interactive click position will be used. If `true` the previous value is irnored and the user is asked to click again.
"""
function find_ft_peak(dat, k_est=nothing; method=:FindZoomFT, interactive=true, overwrite=true, exclude_zero=true, max_range=2.1, scale=40, abs_first=true, roi_size=10)
    fdat = ft(dat)
    k_est = let
        if isnothing(k_est)
            if interactive
                pos = get_positions(fdat, overwrite=overwrite)
                pos[1] .- (size(dat).÷ 2 .+1)
            else
                Float64.(find_max(fdat, exclude_zero=exclude_zero))
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
        return find_ft_iter(fdat, k_est; exclude_zero=exclude_zero, max_range=max_range);
    else
        error("Unknown method for localizing peak. Use :FindZoomFT or :FindIter")
    end
end

"""
    align_stack(dat; refno=1, ref = nothing, damp=0.1, max_freq=0.4, dim=3, method=:FindZoomFT)
    aligns a series of images with respect to a reference image `ref`. 
    If `ref==nothing` this image is extracted from the stack at position `refno`. If `refno==nothing` the central position is used.

# Arguments
+ dat: stack to align slicewise
+ ref: reference image to align to (if `isnothing(ref)==false`)
+ reno: number of reference imag in stack (only if `isnothing(ref)==true`)
+ damp: percentage of outer region to damp
+ max_freq: maximal frequency to consider in the correlation
"""
function align_stack(dat; refno=nothing, ref = nothing, damp=0.1, max_freq=0.4, dim=3, method=:FindZoomFT)
    refno = let
        if isnothing(refno)
            refno=size(dat)[dim] ÷2 +1
        else
            refno
        end
    end
	ref = let
		if isnothing(ref)
			ref = dat[:,:,refno]
		else
			ref
		end
	end
	# damp_edge_outside(ref, damp)
	wh = window_hanning(size(ref), border_in=0.0, border_out=1.0-damp)
	fwin = window_radial_hanning(size(ref), 		 
  		border_in=max_freq*0.8,border_out=max_freq*1.2)
		# rr(size(ref)) .< maxfreq .* size(ref)[1]
	imgs = []
	shifts = []
	fref = fwin .* conj(ft(wh .* (ref .- mean(ref))))
	for n=1:size(dat,dim)
        aslice = dropdims(slice(dat, dim, n), dims=dim)
		cor_ft = ft(wh .* (aslice .- mean(aslice))) .* fref
        myshift = find_ft_peak(cor_ft, method=method, interactive=false)
		# m, p = findmax(abs.(cor))
		# midp = size(cor).÷ 2 .+1
		# myshift = midp .- Tuple(p)
		push!(shifts, myshift)
		push!(imgs, shift(aslice, myshift))
		# push!(imgs, cor_ft)
	end
	cat(imgs...,dims=dim) , shifts
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

function optim_correl(k_est, dat, other=dat)
    mynorm(x) = - all_correl_strengths(x, dat, other)
    #@show k_est
    lower = k_est .- 2.1
    upper = k_est .+ 2.1
    #@show mynorm(k_est)
    mygrad(x) = gradient(mynorm,x)[1]
    #@show mygrad(k_est)
    function g!(G, x)
        G .= mygrad(x)
    end
    od = OnceDifferentiable(mynorm, g!, k_est)
    res = optimize(od,lower, upper, k_est, Fminbox()) # {GradientDescent}
    # res = optimize(mynorm, k_est, LBFGS()) #NelderMead()
    Optim.minimizer(res)
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

function get_subpixel_patch(cor, p_est; scale=10, roi_size=4)
    p_mid = size(cor).÷2 .+ 1
    new_size = min.(roi_size .* scale, size(cor))
    roi = select_region(cor,center=p_mid.+p_est,new_size=new_size)  # (iczt(fc .* exp_ikx(size(cor)[1:2], shift_by=.-p_est), scale)
    # return roi
    fc = ft2d(roi .* window_hanning(size(roi), border_in=0.0))
    # fc = ft2d(roi)
    scale = scale .* ones(ndims(cor))
    roi = iczt(fc, scale, (1,2))
    return roi
end

function get_subpixel_peak(cor, p_est=nothing; exclude_zero=true, scale=10, roi_size=4, abs_first=true, dim=3)
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


function get_subpixel_correl(k_est, dat; other=nothing, psf=psf, upsample=true)
    up, up_other = prepare_correlation(dat; upsample=upsample, other=other, psf=psf)    
    optim_correl(k_est, up, up_other)
end

# hf enhancement for better visualization and peak finding
function beautify(x)
    rr2(size(x)).*sum(abs.(x),dims=3)
end


end # module
