using FindShift
using FourierTools
using View5D, TestImages, Noise, FiniteDifferences, IndexFunArrays, Zygote
using LinearAlgebra, Statistics

function test_peak_find()
        # a test for a pure sine wave
        myk = (0,100.5)
        sz = (512,512)
        mysin = cos.(2pi.*yy(sz) .* myk[2]./sz[2]);
        #@vv ft(mysin)
        abs2_ft_peak(mysin, myk .+ (0.0,-0.5))
        win = collect(window_hanning(size(mysin), border_in=0.0))

        # mynorm(x) = -abs2_ft_peak(mysin .* win, x)
        # tvec = [0.1,-100.8]
        # gradient(mynorm,tvec)[1]
        
        # grad(central_fdm(5, 1),mynorm,tvec)[1]
    
        # dat = mysin; k_est = [0.0,100.5];  max_range=2.1

        p =find_ft_peak(mysin,[0.0,100.5])
    
        mypeak = ft2d(mysin .* window_hanning(size(mysin), border_in=(0,0)))
        res = get_subpixel_peak(mypeak,(0,-100), scale=(10,10))    
end

# @vv psf
function sim_pattern(sz, k, phase)
    return 1.0 .+ cos.(k[1].*xx(sz).+k[2].*yy(sz).+phase)
end


function main()
    cd("C:\\Users\\pi96doc\\Documents\\Programming\\Julia\\FindShift\\examples\\")
    # cd("C:\\Users\\pi96doc\\Documents\\Programming\\Julia\\Development\\")
    # includet("find_random_phase_shifts.jl")
    obj = Float32.(testimage("resolution_test_512.tif"));
    sz=size(obj)
    rmax = 50
    pupil = rr(sz) .< rmax
    h = abs2.(ift(pupil))

    k0 = (1.4,0.1)  # just about visible (42% overlap)
    k0 = (1.6,0.1)  # still visible (34% overlap)
    k0 = (1.8,0.1)  # barely visible (26% overlap)
    truth = k0 .* size(cim) ./ 4 ./ pi
    overlap = (4*rmax - norm(truth)) / (4*rmax)
    # k0 = (0.7, 1.4)
    phase = 0.0 # 2 .*pi.*rand(1,1,10)
    NImgs = 10
    jitter_mag = 3.0
    shifts = [jitter_mag.*(rand(2) .- 0.5) for n=1:NImgs]
    sobj = cat([shift(obj, myshift) for myshift in shifts]...,dims=3)
    img = conv_psf(sobj .* sim_pattern(sz, k0, phase), h)
    # img = conv_psf(sobj , h)
    numphotons=10000
    img .*= numphotons/maximum(img)

    nimg = poisson(img);

    aligned, shifts = align_stack(nimg)
    @vt nimg aligned
    #@vt nimg
    #@vt ft2d(nimg)

    am = sum(aligned, dims=3) ./ size(nimg,3)
    ac = correlate(aligned, psf=h)
    acm = correlate(am, psf=h)
    aligned_s = aligned .- am
    cc = correlate(aligned_s .- mean(aligned .- am), other=am .- mean(am), psf=h*h)
    ccp = correlate(aligned .- am, other=am, psf=h, phase_only=true)
    acc = correlate(am,psf=h)
    cc2 = correlate(ift2d(cc), upsample=false, other=upsample2(am[:,:,1]))

    ccv = correlate(am,other=std(aligned, dims=3))
    # @vt ft2d(aligned .- am) ac (ac .- acm) cc ccp cc2

    # @vt beautify(f2) beautify(ac) beautify(ac .- acm) beautify(cc) beautify(cc2)

    cim = sum(beautify(abs.(cc)),dims=3)[:,:,1]
    cvm = var(beautify(abs.(cc)),dims=3)[:,:,1]
    #@vt cim cvm

    #@vv 
    # k_est = (0,-146)
    # k_est = (0,-114)
    #@vv get_subpixel_patch(cim,k_est, scale=(40,40))
    #@vv get_subpixel_patch(cc, k_est, scale=(40,40))
    
    # res = get_subpixel_peak(cim, scale=(40,40), abs_first=true)
    p =find_ft_peak(cim, method=:FindIter, overwrite=true)
    @show truth
    @show p
    # p =find_ft_peak(ft(cc),[0.3,115.0])

    # res = get_subpixel_peak(cim,(0,0), scale=(10,10))
    
    #@vv get_subpixel_patch(mypeak ,(0,-100), scale=(10,10))


    # mykk = 2pi.* myk./sz
    # dat = 1 .+ mysin
    # abs.(correl_at(myk .+ (-0.0,-0.0), dat))
    # all_correl_strengths(myk.+ (-0.0,-0.0), dat)
    # get_subpixel_correl([0,-100.0].+ [0.0,0.0], dat, other=dat, psf=nothing, upsample=false)
    # all_correl_strengths([0,-200.63], dat)
    # abs.(get_subpixel_correl(myk.+ (-0.0,0.0), dat, other=dat, psf=nothing, upsample=false))

    get_subpixel_correl([0,-114.08].+ [0.0,0.0], nimg .- am, other=am, psf=h, upsample=false)

    @vv correlate(nimg .- am, other=am, psf=h, upsample=false)[1]
    cc, f4 = correlate(nimg .- am, other=am, psf=h, upsample=false)
    @vv cc
end


function test_Optim()
    f(x) = sum(abs2.(x))
    function g!(storage, x)
        storage .= gradient(f,x)[1]
        end

    lower = [-1.25, -2.1]
    upper = [Inf, Inf]
    initial_x = [2.0, 2.0]
    od = OnceDifferentiable(f, g!, initial_x)
    results = optimize(od, lower, upper, initial_x, Fminbox())
end
