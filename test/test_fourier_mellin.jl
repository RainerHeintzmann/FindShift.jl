
function test_fm()
    img1 = Float32.(testimage("resolution"));
    img2 = shift(resample_nfft(rotate(img1, 72.0), t->1.7 .*t), (10.2,52.3));

    img3, param = fourier_mellin(img2, img1)
    # img4, param = fourier_mellin(img3, img1)
    # @vt img1 img2, img3

    nimg1 = poisson(img1, 10)
    nimg2 = poisson(img2, 10)

    nimg3, param = fourier_mellin(nimg2, nimg1)
    # @vt nimg1 nimg2, nimg3

end
