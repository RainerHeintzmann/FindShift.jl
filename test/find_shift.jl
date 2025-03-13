using FindShift
using SeparableFunctions
using FiniteDifferences
using TestImages
using IndexFunArrays
using FourierTools # for ift in psf generation

@testset "Testset find_shift_iter" begin
    sz = (100,103)
    img1 = rand(Float32, sz...);
    myshift = (10.2,22.3)
    img2 = shift(img1, myshift);

    Δx = find_shift_iter(img2, img1)
    @test all(abs.(Δx .- myshift) .< 1e-1) # 1/10th of a pixel precision

    # create a non-commensurate cosine wave and find the peak

    # sz = (100, 103)  
    # k0 = [17.2, 22.3]
    # a = 1.0
    # cos_wave = a .* imag.(exp_ikx_col(sz, shift_by = k0))
    # k, phi, a = FindShift.lsq_fit_sin(cos_wave, k0, 0f0, 1f0)
    # k, phi, a = FindShift.lsq_fit_sin(cos_wave, [17f0, 22f0], 0f0, 1f0)

    sz = (10, 11)  
    vec0 = [1.2, 2.3, 0.0, 1.0]
    cos_wave = vec0[4] .* imag.(exp_ikx_col(sz, shift_by = vec0[1:2]))
    N = prod(sz)*10
    loss = (vec) -> sum(abs2.(vec[4] .* imag.(exp(1im * vec[3]) .* exp_ikx_col(sz, shift_by = vec[1:2])) .- cos_wave) )/N
    # loss = (k) -> sum(abs2.(cos_wave .- a .* imag.(exp_ikx_col(sz, shift_by = k))) )/prod(sz)/100

    # starting values for search:
    vec1 = [1.0, 2.0, 0.0, 1.0]
    mygrad = grad(central_fdm(5, 1), loss, vec1)

    fg! = FindShift.get_sin_fit_fg!(cos_wave)

    G = zeros(Float32, 4)
    myloss = fg!(1, G, vec1)
    @test isapprox(G, mygrad[1], rtol=1e-3)

    k1 = vec1[1:2]
    k, phi, a = FindShift.lsq_fit_sin(cos_wave, k1, 0f0, 1f0)
    @test isapprox([k..., phi, a], vec0, rtol=1e-3)

    sz = (20, 21)  
    # vec0 = [1.0, 2.0, 0.0, 1.0]
    vec0 = [1.2, 2.3, 0.0, 1.0]
    exp_wave = vec0[4] .* exp_ikx_col(sz, shift_by = vec0[1:2])

    N = prod(sz)*10
    loss = (vec) -> sum(abs2.(vec[4] .* exp(1im * vec[3]) .* exp_ikx_col(sz, shift_by = vec[1:2]).- exp_wave) ) / N
    # loss = (k) -> sum(abs2.(cos_wave .- a .* imag.(exp_ikx_col(sz, shift_by = k))) )/prod(sz)/100

    vec1 = [1.0, 2.0, 0.0, 1.5]
    mygrad = grad(central_fdm(5, 1), loss, vec1)

    fg! = FindShift.get_exp_ikx_fit_fg!(exp_wave)

    G = zeros(Float32, 4)
    # G = zeros(ComplexF32, 4)
    myloss = fg!(1, G, vec1)
    @test isapprox(G, mygrad[1], rtol=1e-2)

    k, phi, a = FindShift.lsq_fit_exp_ikx(exp_wave, k1, 0f0, 1f0)
    @test isapprox([k..., phi, a], vec0, rtol=1e-3)

    # true peak is at   vec0[1:2]=[1.2, 2.3]
    k_zoom = find_ft_peak(exp_wave, method=:FindZoomFT)
    @test isapprox([.-k_zoom[1]..., k_zoom[2:3]...], vec0, rtol=2e-2)
    k_iter = find_ft_peak(exp_wave, method=:FindIter)
    @test isapprox([.-k_iter[1]..., k_iter[2:3]...], vec0, rtol=1e-3)
    k_fit = find_ft_peak(exp_wave, method=:FindWaveFit)
    @test isapprox([.-k_fit[1]..., k_fit[2:3]...], vec0, rtol=1e-3)
end

@testset "Correlations" begin
    data = Float32.(testimage("resolution_test_512"))
    sz = size(data)
    psf = abs2.(ift(rr(Float32, sz) .< 100))
    psf = psf / sum(psf)

    vec_c = [1.4, 2.2, 0.0, 1.0]
    cos_modulation = vec_c[4] .* cos.(vec_c[3] .+ vec_c[1].*(1:sz[1]) .+ vec_c[2].*transpose(1:sz[2])) .+ 1

    perfect_data = conv_psf(data .* cos_modulation, psf)
    noisy_data = perfect_data .+ 0.05 .* randn(Float32, sz)
    # test the correlation

    psf_bandpass = nothing # real.(ift(ft(psf) .* (1 .-gaussian_sep(size(psf), sigma=2.0))))
    vec_pos = vec_c[1:2] .* size(psf)/(2pi)
    highpass = rr(size(psf).*2, offset=size(psf) .+ 1 .+ vec_pos) .< 50 # gaussian_sep(size(psf).*2, sigma=55.0)
    # highpass = rr(size(psf).*2) .> 50 # gaussian_sep(size(psf).*2, sigma=55.0)

    corr_res_k, correl_res_p, correl_res_a = get_subpixel_correl(noisy_data; other=nothing, k_est=nothing, psf=psf_bandpass, upsample=true, correl_mask = highpass)
    @test isapprox([corr_res_k[1:2]...], vec_pos, rtol=1e-2)

    # corr_res_k, correl_res_p, correl_res_a = get_subpixel_correl(noisy_data; interactive=true, other=nothing, k_est=nothing, psf=psf_bandpass, upsample=true, correl_mask = highpass)
end

@testset "align_stack" begin
  dat = rand(100,100);

  N=10; sh = [rand(2) for d=1:N]; sh[N÷2+1] = [0.0,0.0]
  dats = cat((shift(dat, .- sh[d]) for d=1:N)..., dims=3);
  dat_aligned, shifts = align_stack(dats);
  @test size(dat_aligned) == size(dats)
  @test size(shifts[1]) == (3,)
  myerr = [abs.(shifts[d][1:2] .- sh[d]) for d in 1:length(sh)]

  @test all(maximum(myerr) .< 0.01)
end
