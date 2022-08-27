@testset "Testset fourier_mellin" begin
    sz = (100,100)
    img1 = zeros(Float32, sz);
    img1[20:80,40:60] .= rand(61,21);
    img1 = FindShift.gaussf(img1);
    α = 22.0 *pi/180
    myzoom = 1.2
    myshift = (10.2,-22.3)
    img2 = shift(resample_nfft(rotate(img1, α), t->myzoom .*t), myshift);

    img3, param = fourier_mellin(img1, img2, radial_exponent=3.0)
    @test abs(param[1] - α) < 0.01
    @test abs(param[2] - myzoom) < 0.1
    @test all(abs.(param[3] .- myshift) .< 0.1)   

    # @vt img1 img2, img3

    # nimg1 = poisson(img1, 10)
    # nimg2 = poisson(img2, 10)
    # nimg3, param = fourier_mellin(nimg2, nimg1)
    # @vt nimg1 nimg2, nimg3

end