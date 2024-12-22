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

res = find_ft_peak(up .* conj.(up_other), (100.0,0.0,0.0); method=:FindPhase, psf=mypsf, verbose=true, ft_mask=nothing, interactive=false, reg_param=1e-16);
res[3]

res = find_ft_peak(up .* conj.(up_other), (210.0,0.0,0.0); method=:FindPhase, psf=mypsf, verbose=true, ft_mask=nothing, interactive=false, reg_param=1e-16);
res[3]

########## more complex example
sz = (1000,1000)
obj = rand(sz...)
k = 100.0
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

# rec_psf = collect(delta(sz))
rec_psf = mypsf
# rec_psf = asf
res = find_ft_peak(wf .* conj.(img_sub), (100.0,0.0,0.0); method=:FindPhase, psf=rec_psf, verbose=true, ft_mask=nothing, interactive=false);
res0 = find_ft_peak(wf .* conj.(wf), (0.0,0.0,0.0); method=:FindPhase, psf=rec_psf, verbose=true, ft_mask=nothing, interactive=false);
res[3]/res0[3]

mydelta = collect(delta(sz))
# rec_psf = mydelta
rec_psf = sim_psf # important that it is normalized!
# rec_psf = asf
rec_psf2 = conv_psf(sim_psf,sim_psf)
res = get_subpixel_correl(wf;  other=img_sub, k_est=(100.0, 0.0, 0.0), psf=rec_psf, upsample=false, interactive=false, correl_mask=nothing, method=:FindPhase, verbose=false, reg_param=1e-6)
# the factor is correl / sum(wf * otf * conj(wf) * otf_shifted). Since wf already contains an OTF, the product cancels out.
# We only need to account for the fact that the OTF is shifted via the other_psf_argument. This shifting is performed via k_other_psf_shift.
res0 = get_subpixel_correl(wf; k_est=(0.0, 0.0, 0.0), k_other_psf_shift = (100.0, 0.0, 0.0), psf=nothing, other_psf=rec_psf2, upsample=false, interactive=false, correl_mask=nothing, method=:FindPhase, verbose=false, reg_param=1e-6)
res[3]/res0[3]

abs(get_rel_subpixel_correl(wf, img_sub, (100.0, 0.0, 0.0), rec_psf; upsample=false))
# @vt ft(conv_psf(img_sub,rec_psf) .* conj.(conv_psf(wf, rec_psf))) ft(conv_psf(wf,rec_psf) .* conj.(conv_psf(wf, rec_psf)))
# @vt ft(img_sub .* conj.(wf .- mean(wf))) ft(abs2.(wf)) ft(img_sub)
