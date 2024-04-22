@testset "Testset fourier_mellin" begin
    img1 = ones(Float32, 100,100) # rand(100,100)
    α = 30.0 *pi/180
    zoom = 2.0
    ϕ = FindShift.get_rigid_warp((α, zoom, [0.0, 0.0]), size(img1))

    @test ϕ((51,51)) ≈ [51,51]
    # @test ϕ((55,51)) ≈ [57.92829323927,55]

    # res = replace_nan(warp(img1, ϕ, size(img1)))
    ns = Integer.(1.2.*size(img1))
    res = FindShift.apply_warp(img1, ϕ, ns)
    # res2 = warp(img1, ϕ, ns, fillvalue=0f0)

    @test eltype(res) == Float32
    @test res[33,33] == 1.0
    @test isnan(res[32,32])
    res = FindShift.apply_warp(img1, ϕ, ns, fillvalue=0f0)
    @test res[32,32] == 0.0
end
