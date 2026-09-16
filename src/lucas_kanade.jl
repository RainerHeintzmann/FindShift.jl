"""
    find_shift_lk(dat1::AbstractArray{T,N}, dat2::AbstractArray{T,N}, Δx=nothing; maxiter=10, tol=1e-4, smoothing=1.0) where {T,N}

finds the sub-pixel shift between two same-sized patches `dat1` and `dat2` using the classical
Lucas-Kanade optical-flow method (image-gradient-based Gauss-Newton translation estimation),
as an alternative to the Fourier/Optim-based [`find_shift_iter`](@ref).

`dat1` is treated as the fixed template and `dat2` as the moving patch, following the same convention
as [`find_shift_iter`](@ref): the returned shift `Δx` is such that `shift(dat2, Δx) ≈ dat1`. `dat1` and
`dat2` need not have the same size (e.g. `dat2` may be a larger search region around `dat1`, as used by
[`find_deformations`](@ref)) -- as in [`find_shift_iter`](@ref), both are cropped to a common size around
the initial integer estimate via [`shift_cut`](@ref) before the sub-pixel refinement.

As with [`find_shift_iter`](@ref), an initial integer-pixel estimate is first obtained via the FFT
cross-correlation in [`find_shift`](@ref) (which works for arbitrarily large shifts), and the
Lucas-Kanade Gauss-Newton iteration then only has to resolve the remaining sub-pixel residual, for
which it converges in only a handful of iterations. The patches are Gaussian pre-smoothed (`smoothing`)
before the image gradients are computed, since Lucas-Kanade relies on locally well-defined intensity
gradients and is otherwise unreliable on noisy or high-frequency content.

# Arguments
+ `dat1`: fixed/reference patch
+ `dat2`: moving patch, to determine the shift for (relative to `dat1`)
+ `Δx`: an initial shift estimate. `nothing` (default) determines one via [`find_shift`](@ref) first.
+ `maxiter`: maximum number of Gauss-Newton iterations
+ `tol`: convergence tolerance (in pixels) on the update-step norm
+ `smoothing`: standard deviation (in pixels) of the Gaussian pre-smoothing applied before computing gradients

# Example
```julia
julia> img = rand(100,100);

julia> simg = shift(img, (3.3, 4.45));

julia> find_shift_lk(simg, img)
2-element Vector{Float64}:
 3.29...
 4.44...
```
"""
function find_shift_lk(dat1::AbstractArray{T,N}, dat2::AbstractArray{T,N}, Δx=nothing; maxiter=20, tol=1e-4, smoothing=1.0) where {T,N}
    RT = real(T)
    Δx = let
        if isnothing(Δx)
            find_shift(dat1, dat2) # to get a first (integer) estimate, as a Tuple
        else
            Δx
        end
    end

    # crop dat1 and dat2 (which may differ in size) to a common size around the integer estimate,
    # so only the (small) sub-pixel residual remains to be resolved by the Gauss-Newton iteration.
    dat1c, dat2c = shift_cut(dat1, dat2, .-Δx)
    d1 = filter_gaussian(RT.(dat1c), RT(smoothing))
    d2 = filter_gaussian(RT.(dat2c), RT(smoothing))
    win = window_hanning(RT, size(d1), border_in=0.0)
    wd1 = win .* d1
    sub = zeros(RT, N)
    converged = false
    for _ in 1:maxiter
        shifted = shift(d2, sub)
        δ = lk_step(wd1, shifted, win)
        sub = sub .+ δ
        if norm(δ) < tol
            converged = true
            break
        end
    end
    # note: `sub` is allowed to end up far from the initial integer estimate `Δx` -- the FFT-correlation
    # based integer bootstrap can itself be off by a pixel or more on smooth/low-texture content, and the
    # Gauss-Newton iteration is expected to (and does) correct for that. Only a failure to converge at all
    # indicates an actual problem.
    if !converged
        @warn "Lucas-Kanade sub-pixel refinement did not converge within $maxiter iterations. Using integer correlation estimate instead."
        return Δx
    end
    return Δx .+ sub
end

"""
    lk_step(template, shifted, win)

single Gauss-Newton Lucas-Kanade update: given the (already windowed) `template`, the current
un-windowed shifted estimate `shifted` of the moving patch, and the window `win`, returns the
additive correction `δ` to the shift so that `shift(shifted, δ)` gets closer to `template`.

The spatial gradient used for the linearization is computed from the *un-windowed* `shifted` array
and only weighted by `win` afterwards -- computing it directly on `win .* shifted` would incorrectly
mix in the window's own slope and bias/slow down the convergence. Used internally by [`find_shift_lk`](@ref).
"""
function lk_step(template::AbstractArray{T,N}, shifted::AbstractArray{T,N}, win) where {T,N}
    RT = real(T)
    raw_grads = ntuple(N) do d
        (circshift(shifted, ntuple(i -> i == d ? -1 : 0, N)) .-
         circshift(shifted, ntuple(i -> i == d ? 1 : 0, N))) ./ 2
    end
    grads = ntuple(d -> win .* raw_grads[d], N)
    err = template .- win .* shifted
    A = zeros(RT, N, N)
    b = zeros(RT, N)
    for i in 1:N
        b[i] = -sum(real.(grads[i] .* conj.(err)))
        for j in i:N
            A[i, j] = sum(real.(grads[i] .* conj.(grads[j])))
            A[j, i] = A[i, j]
        end
    end
    reg = (tr(A) / N) * RT(1e-6) + eps(RT)
    return (A + reg * I) \ b
end
