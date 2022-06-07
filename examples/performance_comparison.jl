using FindShift
using FourierTools
using View5D, TestImages, Noise, FiniteDifferences, IndexFunArrays, Zygote
using LinearAlgebra, Statistics
using Random
using View5D # for get_positions

export peak_compare

function peak_compare(myk = nothing;method=:FindIter)
        # a test for a pure sine wave        
        myk = let
            if isnothing(myk)
                200.0 .* (rand(2) .- 0.5)
            else
                myk
            end
        end
        sz = (512,512)
        mysin = exp.(2pi.*1im.*(xx(sz) .* myk[1]./sz[1] .+ yy(sz) .* myk[2]./sz[2]));
        #@vv ft(mysin)
        # abs2_ft_peak(mysin, myk .+ (0.0,-0.5))
        win = collect(window_hanning(size(mysin), border_in=0.0)) # helps a bit. 

        p =find_ft_peak(win .* mysin, interactive=false, method=method, abs_first=false, scale=60) # abs_first=true has a problem!
        return myk, myk .- p
end

function test_iter()
    myk = [10.2, 22.3]
    sz = (512,512)
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

function compare_performance_cos(N=20)

        Random.seed!(3)
        myerr = []
        for n=1:N
            # pos, err = peak_compare(method=:FindZoomFT)  # 0.011349
            pos, err = peak_compare(method=:FindIter)
            push!(myerr, err)
        end

        mystd = mean(std(myerr)) 

        mypeak = ft2d(mysin .* window_hanning(size(mysin), border_in=(0,0)))
        res = get_subpixel_peak(mypeak,(0,-100), scale=(10,10))    
end
