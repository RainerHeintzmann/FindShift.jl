using FindShift
using SeparableFunctions
using FiniteDifferences
using BenchmarkTools

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
k_zooms = []; k_iters = []; k_fits = []
p_zooms = []; p_iters = []; p_fits = []
a_zooms = []; a_iters = []; a_fits = []
N = 100
StdNoise = 50.0
for n=1:N
    noisy_exp_wave = exp_wave .+ StdNoise .* randn(ComplexF32, sz)
    k, p, a  = find_ft_peak(noisy_exp_wave, method=:FindZoomFT) 
    push!(k_zooms, k); push!(p_zooms, p); push!(a_zooms, a)
    k, p, a  = find_ft_peak(noisy_exp_wave, method=:FindIter) 
    push!(k_iters, k); push!(p_iters, p); push!(a_iters, a)
    k, p, a = find_ft_peak(noisy_exp_wave, method=:FindWaveFit) 
    push!(k_fits, k); push!(p_fits, p); push!(a_fits, a)
    print("$(n)/$(N), ")
end
k_zooms = hcat(k_zooms...); k_iters = hcat(k_iters...); k_fits = hcat(k_fits...)
print("ZoomFT: $(mean(k_zooms,dims=2)) ± $(std(k_zooms, dims=2))\n")
print("Iter: $(mean(k_iters,dims=2)) ± $(std(k_iters, dims=2))\n")
print("WaveFit: $(mean(k_fits,dims=2)) ± $(std(k_fits, dims=2))\n") # most accurate

print("ZoomFT: $(mean(p_zooms)) ± $(std(p_zooms))\n")
print("Iter: $(mean(p_iters)) ± $(std(p_iters))\n")
print("WaveFit: $(mean(p_fits)) ± $(std(p_fits))\n")

print("ZoomFT: $(mean(a_zooms)) ± $(std(a_zooms))\n")
print("Iter: $(mean(a_iters)) ± $(std(a_iters))\n")
print("WaveFit: $(mean(a_fits)) ± $(std(a_fits))\n")

# @vtp ft(noisy_exp_wave) noisy_exp_wave
