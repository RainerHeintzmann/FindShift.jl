@testset "Testset find_shift_iter" begin
    sz = (100,100)
    img1 = rand(Float32, sz...);
    myshift = (10.2,22.3)
    img2 = shift(img1, myshift);

    Δx = find_shift_iter(img2, img1)
    @test all(abs.(Δx .- myshift) .< 1e-1) # 1/10th of a pixel precision
end