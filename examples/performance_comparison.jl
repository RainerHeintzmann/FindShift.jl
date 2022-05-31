using FindShift
using FourierTools
using View5D, TestImages, Noise, FiniteDifferences, IndexFunArrays, Zygote
using LinearAlgebra, Statistics
using Random

export sim_compare

function sim_compare(;method=:FindIter)
        # a test for a pure sine wave
        myk = 200.0 .* (rand(2) .- 0.5)
        sz = (512,512)
        mysin = exp.(2pi.*1im.*(xx(sz) .* myk[1]./sz[1] .+ yy(sz) .* myk[2]./sz[2]));
        #@vv ft(mysin)
        # abs2_ft_peak(mysin, myk .+ (0.0,-0.5))
        win = collect(window_hanning(size(mysin), border_in=0.0)) # helps a bit. 

        p =find_ft_peak(win .* mysin, interactive=false, method=method, abs_first=false, scale=60) # abs_first=true has a problem!
        return myk, myk .- p
end

function compare_performance_cos(N=20)

        Random.seed!(3)
        myerr = []
        for n=1:N
            pos, err = sim_compare(method=:FindZoomFT)  # 0.011349
            # pos, err = sim_compare(method=:FindIter)
            push!(myerr, err)
        end

        mystd = mean(std(myerr)) 

        mypeak = ft2d(mysin .* window_hanning(size(mysin), border_in=(0,0)))
        res = get_subpixel_peak(mypeak,(0,-100), scale=(10,10))    
end
