"""
    extract_sub_images(images; corners=nothing, extract_size=nothing, images_per_image=[2,1], flip_even = true)

rearranges multiple images recorded on a single camera into a list of images.
#Arguments
+ `corners`:    a list of `tuple` corner postions in each of the raw data input images. If `nothing` they will be calculated according to geometry
+ `extract_size`:   the `tuple` size of each sub-image to extract
+ `images_per_image`: spefies the arrangement of sub-images. Default is a 2x1 arrangement meaning two aling the x coordingate
+ `flip_even`:  determines that every other input-image needs to be flipped
"""
function extract_sub_images(images; corners=nothing, extract_size=nothing, images_per_image=(2,1), flip_even = true)
    if isnothing(extract_size)
        extract_size = Tuple(size(images[1]) .÷ images_per_image)
    end
    extracted = []
    n = 1
    ni = 1
    for img in images
        for patch in CartesianIndices(images_per_image)
            if isnothing(corners)
                @show patch
                @show pos = Tuple(round.(Int, (Tuple(patch) .- 0.5).*size(img)./images_per_image))
                ex = select_region(img, center=pos, new_size= extract_size)
            else
                ex = img[corners[n,1]:corners[n,1]+extract_size[1]-1,corners[n,2]:corners[n,2]+extract_size[2]-1];
            end
            if flip_even && iseven(ni)
                ex = reverse(ex, dims=1)
            end
            push!(extracted, ex)
            n += 1
        end
        ni += 1
    end
    return extracted
end
