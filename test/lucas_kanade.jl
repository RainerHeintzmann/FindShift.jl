using FindShift
using FourierTools

@testset "find_shift_lk" begin
    sz = (100, 103)
    img1 = rand(Float32, sz...)
    myshift = (3.3, -4.45)
    img2 = shift(img1, myshift)

    Δx = find_shift_lk(img2, img1)
    @test all(abs.(Δx .- myshift) .< 1e-1) # 1/10th of a pixel precision

    # a larger shift, exercising the coarse-to-fine pyramid
    myshift_big = (10.2, 22.3)
    img3 = shift(img1, myshift_big)
    Δx_big = find_shift_lk(img3, img1)
    @test all(abs.(Δx_big .- myshift_big) .< 2e-1)
end
