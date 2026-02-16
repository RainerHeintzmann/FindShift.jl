using FindShift
using SeparableFunctions
using FiniteDifferences
using BenchmarkTools
using IndexFunArrays # for Hann window
using TestImages
using FourierTools
using Noise
using NDTools # for idx_to_dim

# benchmark
sz = (128, 128)  
# sz = (2048, 2048)  
vec0 = ([11.2, 14.3], 0.3, 1.0) # position, phase, amplitude

# First: perfect exponential wave
exp_wave = vec0[3] .* exp(1im*vec0[2]) .* exp_ikx_col(sz, shift_by = .-vec0[1])
# exp_wave .*= (gaussian_sep(sz, sigma=(3.0, 4.0), offset=(33.5, 34.2)) .+  # finds the wrong intitial guess! Only :FindZoomFT works here.
#             gaussian_sep(sz, sigma=(3.0, 4.0), offset=(44.5, 14.2)) .+
#             gaussian_sep(sz, sigma=(3.0, 4.0), offset=(44.5, 114.2)))
exp_wave .*= (gaussian_sep(sz, sigma=(3.0, 4.0), offset=(33.5, 34.2)) .+
             gaussian_sep(sz, sigma=(3.0, 4.0), offset=(44.5, 14.2))); # NOT WORKING FOR :FindShiftedWindow
exp_wave .*= gaussian_sep(sz, sigma=(3.0, 4.0), offset=(33.5, 34.2)); # ALSO WORKING FOR :FindShiftedWindow
vec_zoom = find_ft_peak(exp_wave, method=:FindZoomFT)[1] .- vec0[1] # 0,0
vec_iter = find_ft_peak(exp_wave, method=:FindIter)[1] .- vec0[1]  # -2e-7, 1e-6
vec_fit = find_ft_peak(exp_wave, method=:FindWaveFit)[1] .- vec0[1] # -8e-7, 3e-7
vec_com = find_ft_peak(exp_wave, method=:FindCOM)[1] .- vec0[1] # -0.25, -0.11 BAD FOR SHARP PEAKS
vec_par = find_ft_peak(exp_wave, method=:FindParabola)[1] .- vec0[1] # -0.16, -0.09 BAD FOR SHARP PEAKS
vec_shift = find_ft_peak(exp_wave, method=:FindShiftedWindow)[1] .- vec0[1] # -0.01, -0.01 BAD FOR BROAD PEAKS

@btime k_zoom = find_ft_peak($exp_wave, method=:FindZoomFT) # 6.9 ms, 125 ms
@btime k_iter = find_ft_peak($exp_wave, method=:FindIter) # 2.8 ms, 1.437 s
@btime k_fit = find_ft_peak($exp_wave, method=:FindWaveFit) # 5.5 ms, 3.99 s
@btime k_com = find_ft_peak($exp_wave, method=:FindCOM) # 0.27 ms, 170 ms
@btime k_par = find_ft_peak($exp_wave, method=:FindParabola) # 0.270 ms,  189 ms
@btime k_shift = find_ft_peak($exp_wave, method=:FindShiftedWindow) # 0.357 ms,  190 ms

sz = (1024, 1024)  
exp_wave = vec0[4] .* exp_ikx_col(sz, shift_by = vec0[1:2])
@btime k_zoom = find_ft_peak($exp_wave, method=:FindZoomFT) # 41 ms
@btime k_iter = find_ft_peak($exp_wave, method=:FindIter) # 332 ms
@btime k_fit = find_ft_peak($exp_wave, method=:FindWaveFit) # 888 ms

# noise sensitivity
sz = (512, 512)  
exp_wave = vec0[3] .* exp(1im*vec0[2]) .* exp_ikx_col(sz, shift_by = .-vec0[1])
hw = window_hanning(real.(exp_wave), border_in=0.0)

VT = Vector{Tuple{Float64, Float64}}
k_zooms = VT(); k_iters = VT(); k_fits = VT(); k_fits_po = VT()
p_zooms = []; p_iters = []; p_fits = []; p_fits_po = []
a_zooms = []; a_iters = []; a_fits = []; a_fits_po = []
N = 100
StdNoise = 50.0
use_win = false
for n=1:N
    noisy_exp_wave = exp_wave .+ StdNoise .* randn(ComplexF32, sz)
    if (use_win)
        noisy_exp_wave .*= hw
    end
    k, p, a  = find_ft_peak(noisy_exp_wave, method=:FindZoomFT) 
    push!(k_zooms, Tuple(k)); push!(p_zooms, p); push!(a_zooms, a)
    k, p, a  = find_ft_peak(noisy_exp_wave, method=:FindIter) 
    push!(k_iters, Tuple(k)); push!(p_iters, p); push!(a_iters, a)
    k, p, a = find_ft_peak(noisy_exp_wave, method=:FindWaveFit) 
    push!(k_fits, Tuple(k)); push!(p_fits, p); push!(a_fits, a)
    k, p, a = find_ft_peak(noisy_exp_wave, method=:FindWaveFit, phase_only=true) 
    push!(k_fits_po, Tuple(k)); push!(p_fits_po, p); push!(a_fits_po, a)
    print("$(n)/$(N), ")
end
k_zoom_m = idx_to_dim(k_zooms); k_iter_m = idx_to_dim(k_iters); 
k_fit_m = idx_to_dim(k_fits); k_fit_po_m = idx_to_dim(k_fits_po)
std_diff(myks, myvec) = sqrt.(mean(abs2.(myks .- (myvec[1])'), dims=1)) 

print("ZoomFT: $(mean(k_zoom_m,dims=1)) ± $(std(k_zoom_m, dims=1)), diff: $(std_diff(k_zoom_m, vec0))\n") # 0.036
print("Iter: $(mean(k_iter_m,dims=1)) ± $(std(k_iter_m, dims=1)), diff: $(std_diff(k_iter_m, vec0))\n") # 0.066
print("WaveFit: $(mean(k_fit_m,dims=1)) ± $(std(k_fit_m, dims=1)), diff: $(std_diff(k_fit_m, vec0))\n") # 0.035 most accurate
print("WaveFit Phase only: $(mean(k_fit_po_m,dims=1)) ± $(std(k_fit_po_m, dims=1)), diff: $(std_diff(k_fit_po_m, vec0))\n") # 0.042 

print("ZoomFT: $(mean(p_zooms)) ± $(std(p_zooms))\n")
print("Iter: $(mean(p_iters)) ± $(std(p_iters))\n")
print("WaveFit: $(mean(p_fits)) ± $(std(p_fits))\n")

print("ZoomFT: $(mean(a_zooms)) ± $(std(a_zooms))\n")
print("Iter: $(mean(a_iters)) ± $(std(a_iters))\n")
print("WaveFit: $(mean(a_fits)) ± $(std(a_fits))\n")

# @vtp ft(noisy_exp_wave) noisy_exp_wave

# Now try to find the shift between images:

obj = Float32.(testimage("resolution_test_512"))
sz = size(obj)
psf = abs2.(ift(rr(sz) .> size(obj,1)/4.5))
pimg = conv_psf(obj .+ 0.025, psf) # constant added to avoid negative values
nphotons = 5.0
pimg .*= nphotons/maximum(pimg)

vec0 = ([11.2, 14.3], 0.3, 1.0) # position, phase, amplitude

simg = shift(pimg, .-vec0[1])

hw = window_hanning(obj, border_in=0.8)

VT = Vector{Tuple{Float64, Float64}}
all_sz = VT(); all_si = VT(); all_smz = VT(); all_smi = VT(); all_smip = VT(); all_w = VT();  
N = 100
StdNoise = 50.0
use_win = true
use_poisson = true
use_anscombe = true
use_phaseonly = false
use_upsample = false
# anscombe(dat) = 2 .*sqrt.(dat .+ eltype(dat)(3/8));
anscombe(dat) = 2 .*sqrt.(dat); # important: Do NOT apply the 3/8 here, as this leads to bad results!
for n=1:N
    if (use_poisson)
        nimg = Float32.(Noise.poisson(pimg))
        nsimg = Float32.(Noise.poisson(simg).* 1.5) # to see if wrong scaing has an influence
    else
        nimg = pimg .+ StdNoise .* randn(ComplexF32, sz)
        nsimg = simg .+ StdNoise .* randn(ComplexF32, sz)
    end
    if (use_anscombe)
        nimg = anscombe(nimg)
        nsimg = anscombe(nsimg)
    end
    if (use_win)
        nimg .*= hw
        nsimg .*= hw
    end

    sz  = find_shift(nimg, nsimg; zoom=100) # method=:FindZoomFT
    push!(all_sz, sz)
    si  = find_shift_iter(nimg, nsimg)
    push!(all_si, Tuple(si))
    s_mz = get_subpixel_correl(ft(nsimg);  other=ft(nimg), psf=nothing, 
                            upsample=use_upsample, correl_mask=nothing, 
                            method=:FindZoomFT, phase_only=use_phaseonly, scale=80)[1][1:2]
    push!(all_smz, s_mz)
    s_mi = get_subpixel_correl(ft(nsimg);  other=ft(nimg), psf=nothing, 
                            upsample=use_upsample, correl_mask=nothing, 
                            method=:FindIter, phase_only=use_phaseonly)[1][1:2]
    push!(all_smi, Tuple(s_mi))
    s_w = get_subpixel_correl(ft(nsimg);  other=ft(nimg), psf=nothing, 
                            upsample=use_upsample, correl_mask=nothing, 
                            method=:FindWaveFit, phase_only=use_phaseonly)[1][1:2]
    push!(all_w, Tuple(s_w))
    ncsimg = conv_psf(nsimg, psf)
    ncimg = conv_psf(nimg, psf)
    # Careful: PSF is convolved starting in Fourier space, so this accounts rather for different signal strengths
    s_mip = get_subpixel_correl(collect(ft(ncsimg));  other=collect(ft(ncimg)), psf=nothing, 
                            upsample=use_upsample, correl_mask=nothing, 
                            method=:FindIter, phase_only=use_phaseonly)[1][1:2]
    push!(all_smip, Tuple(s_mip))

    print("$(n)/$(N), ")
end
sz_m = idx_to_dim(all_sz); 
si_m = idx_to_dim(all_si); 
smz_m = idx_to_dim(all_smz); 
smi_m = idx_to_dim(all_smi); 
smip_m = idx_to_dim(all_smip); 
w_m = idx_to_dim(all_w); 

std_diff(myks, myvec) = sqrt.(mean(abs2.(myks .- (myvec[1])'), dims=1)) 

print("ZoomFT1: $(mean(sz_m,dims=1)) ± $(std(sz_m, dims=1)), diff: $(std_diff(sz_m, vec0))\n") # 3.9% with Anscombe
print("Iter: $(mean(si_m,dims=1)) ± $(std(si_m, dims=1)), diff: $(std_diff(si_m, vec0))\n") # 
print("ZoomFT2: $(mean(smz_m,dims=1)) ± $(std(smz_m, dims=1)), diff: $(std_diff(smz_m, vec0))\n") # 
print("Iter2: $(mean(smi_m,dims=1)) ± $(std(smi_m, dims=1)), diff: $(std_diff(smi_m, vec0))\n") # 1.3%, 1.0% with Anscombe
print("Iter+PSF: $(mean(smip_m,dims=1)) ± $(std(smip_m, dims=1)), diff: $(std_diff(smip_m, vec0))\n") # 1.3%, 1.0% with Anscombe
print("WaveFit: $(mean(w_m,dims=1)) ± $(std(w_m, dims=1)), diff: $(std_diff(w_m, vec0))\n") # 
# conclusion: The best is get_subpixel_correl with Iter2 with Anscombe (no 3/8), preconvoling does not help, Hanning does not hurt much
# phase_only is bad! Anscombe is pretty robust against scaling issues
# no upsampling is fine!

