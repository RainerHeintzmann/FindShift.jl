using FindShift
using FourierTools
using View5D, TestImages, Noise, FiniteDifferences, IndexFunArrays, Zygote
using LinearAlgebra, Statistics
using Random, NDTools
using View5D # for get_positions

export peak_compare

function peak_compare(myk = nothing; obj = nothing, method=:FindIter, NPhot =100)
        # a test for a pure sine wave        
        myk = let
            if isnothing(myk)
                100.0 .* rand(2)  #  .- 0.5
            else
                myk
            end
        end
        sz = let 
            if isnothing(obj)
                (512,512)
            else
                size(obj)
            end            
        end
        mysin = exp.(2pi.*1im.*(xx(sz) .* myk[1]./sz[1] .+ yy(sz) .* myk[2]./sz[2]));
        win = collect(window_hanning(sz, border_in=0.0)) # helps a bit. 
        #@vv ft(mysin)
        # abs2_ft_peak(mysin, myk .+ (0.0,-0.5))
        if isnothing(obj)
            dat = 1.0 .+ real.(mysin)
            fak =  NPhot ./ maximum(dat)
            ndat = poisson(dat .* fak)
            p =find_ft_peak(win .* (ndat .- mean(ndat)), interactive=false, method=method, exclude_zero=true, abs_first=false, scale=60) # abs_first=true has a problem!
        else
            dat = obj .* (1.0 .+ real.(mysin))
            fak =  NPhot ./ maximum(dat)
            ndat = poisson(dat .* fak)
            wdat = win .* (ndat .- fak .* obj)
            p =optim_correl(wdat, obj .- mean(obj), method=method, verbose=false) # abs_first=true has a problem!
        end
        return myk, myk .- abs.(p)
end

function test_iter()
    myk = [10.2, 22.3]
    sz = (512,512)
    myk = [2.1, 1.2]
    sz = (150,150)
    mysin = exp.(2pi.*1im.*(xx(sz) .* myk[1]./sz[1] .+ yy(sz) .* myk[2]./sz[2]));
    # peak_compare(myk, method=:FindIter)
    win = collect(window_hanning(size(mysin), border_in=0.0)) # helps a bit. 
    dat = win .* mysin
    p =find_ft_peak(dat, interactive=false, method=:FindIter, abs_first=false, scale=60, max_range=2.0) # abs_first=true has a problem!    

    k_est = [Float64.(FindShift.find_max(ft(dat), exclude_zero=false))...]
    v_init = sqrt(abs2_ft_peak(dat, k_est))
    dat = dat ./v_init
    abs2_ft_peak(dat, k_est)
    abs2_ft_peak(dat, myk)
    gradient(abs2_ft_peak, dat, k_est)[2]
    gradient(abs2_ft_peak, dat, myk)[2]        
    # SLOW!!! grad(central_fdm(5, 1),abs2_ft_peak,dat, myk)[2]
    mynorm(x) = -abs2_ft_peak(dat, x)
    gradient(mynorm, k_est)[1]
    grad(central_fdm(5, 1),mynorm, k_est)[1]
    k2 = [11.2,55.4]
    gradient(mynorm, k2)[1]
    grad(central_fdm(5, 1),mynorm, k2)[1]

    gradient(mynorm, myk)[1]
    gradient(mynorm, p)[1]

    function g!(G, x)  # (G, x)
        G .= gradient(mynorm,x)[1]
    end

    od = OnceDifferentiable(mynorm, g!, k_est)
    @time res=optimize(od, k_est, LBFGS(), Optim.Options(show_trace=true, g_tol=1e-3)) #NelderMead(), iterations=2, x_tol = 1e-2, g_tol=1e-3
    res.minimizer

    FindShift.find_ft_iter(dat, k_est; exclude_zero=true, max_range=nothing)
    FindShift.find_ft_peak(dat, k_est; method=:FindIter, exclude_zero=true, max_range=nothing)

    x = 9:0.01:11
    f(x) = mynorm([x,22.3])
    g(x) = mynorm([10.2,x])
    plot(x,f.(x),label="x")
    x1 = 21:0.01:23
    plot!(x1,g.(x1),label="y")

    ff(x) = gradient(mynorm, [x, 22.3])[1][1]
    gg(x) = gradient(mynorm, [10.2, x])[1][2]
    gg2(x) = grad(central_fdm(5, 1), mynorm, [10.2, x])[1][2]
    plot(x,ff.(x), label="dL/dx")
    plot!(x1,gg.(x1), label="dL/dy")
    plot!(x1,gg2.(x1), label="dL/dy finite diff")
end

function compare_performance_cos()
        myerr = []
        myerr_iter = []
        NPhot = 500000
        N = 20
        obj = Float32.(testimage("resolution_test_512.tif"));
        for n=1:N
            Random.seed!(n)
            pos, err = peak_compare(method=:FindZoomFT, NPhot=NPhot, obj=obj)  # 0.0044
            push!(myerr, err)
            Random.seed!(n)
            pos, err = peak_compare(method=:FindIter, NPhot=NPhot, obj=obj) # 5.64 e-5
            push!(myerr_iter, err)
        end

        mystd = mean(std(myerr)) 
        mystd_iter = mean(std(myerr_iter)) 

        mypeak = ft2d(mysin .* window_hanning(size(mysin), border_in=(0,0)))
        res = get_subpixel_peak(mypeak,(0,-100), scale=(10,10))    
end

function grad_checks()
    dat1 = rand(ComplexF64, 5,5)
    dat2 = rand(ComplexF64,5,5)
    f(x) = sum(abs2.(dat1 .- FindShift.exp_shift_dat(dat2,x)))
    # f(x) = sum(abs2.(FindShift.exp_shift_dat(dat,x))) # yields zero
    gradient(f,[0.1,0.2])[1]
    grad(central_fdm(5, 1),f,[0.1,0.2])[1]
end


function test_find_shift()
    obj = Float32.(testimage("resolution_test_512.tif"));
    sz = (170,170)
    dat1 = select_region(obj, new_size=sz)
    dat2 = select_region(shift(obj, (5.5, -7.7)), new_size=sz)

    sv = find_shift_iter(dat1, dat2)
    @vt dat1 dat2 shift(dat2, sv)
end
