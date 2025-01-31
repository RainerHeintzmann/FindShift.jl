using IndexFunArrays
using FourierTools

sz = (1000,1000)
big_pupil = collect(disc(sz, 90));
asf = real.(ift(big_pupil))

pupil = collect(disc(sz, 50));
mypsf = abs2.(ift(pupil));
mypsf = mypsf ./ sum(mypsf)
otf = ft(mypsf)

up = collect(ift(otf));
up_other = up;

res = find_ft_peak(up .* conj.(up_other), (100.0,0.0); method=:FindZoomFT, psf=mypsf, verbose=true, ft_mask=nothing, interactive=false, reg_param=1e-16);
res[3]

res = find_ft_peak(up .* conj.(up_other), (210.0,0.0); method=:FindShiftedWindow, psf=mypsf, verbose=true, ft_mask=nothing, interactive=false, reg_param=1e-16);
res[3]

########## more complex example
sz = (1000,1000)
obj = rand(sz...)
k = 100.234
illu = 1 .+ cos.(2π * (1:sz[1])/sz[1] * k .+ 0.1)

obj = rand(sz...)
sim_psf = asf
sim_psf = mypsf
# sim_psf = collect(delta(sz))

img = conv_psf(obj .* illu, sim_psf)
wf = conv_psf(obj, sim_psf)
img_sub = img .- wf
@vt img img_sub
@vt ft(img) ft(img_sub)

res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindZoomFT)
res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindIter)
res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindWaveFit)
res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindCOM)
res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindParabola)
res = get_subpixel_correl(wf; other= img_sub, k_est=(100.0, 0.0), psf= sim_psf, upsample=false, method=:FindShiftedWindow)

mydelta = collect(delta(sz))
# rec_psf = mydelta
rec_psf = sim_psf # important that it is normalized!
# rec_psf = asf

abs(get_rel_subpixel_correl(wf, img_sub, (100.0, 0.0, 0.0), rec_psf; upsample=false))
# @vt ft(conv_psf(img_sub,rec_psf) .* conj.(conv_psf(wf, rec_psf))) ft(conv_psf(wf,rec_psf) .* conj.(conv_psf(wf, rec_psf)))
# @vt ft(img_sub .* conj.(wf .- mean(wf))) ft(abs2.(wf)) ft(img_sub)
