using FindShift
using SeparableFunctions
using FiniteDifferences

# benchmark
sz = (128, 128)  
sz = (2048, 2048)  
vec0 = ([11.2, 14.3], 0.3, 1.0)

exp_wave = vec0[3] .* exp(1im*vec0[2]) .* exp_ikx_col(sz, shift_by = .-vec0[1])
vec_zoom = find_ft_peak(exp_wave, method=:FindZoomFT) # 7.4 ms
vec_iter = find_ft_peak(exp_wave, method=:FindIter) # 2.5 ms
vec_fit = find_ft_peak(exp_wave, method=:FindWaveFit) # 6.649 ms

@btime k_zoom = find_ft_peak($exp_wave, method=:FindZoomFT) # 7.3 ms
@btime k_iter = find_ft_peak($exp_wave, method=:FindIter) # 2.5 ms
@btime k_fit = find_ft_peak($exp_wave, method=:FindWaveFit) # 6.649 ms

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
