# since Zygote cannot deal with exp_ikx from the IndexFunArray Toolbox, here is an alternative
function exp_shift(sz, k_0)
    # mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    fct = (p, sz, pvec) -> cis(2pi * pvec * p)
    f_sep = calculate_separables(Array{ComplexF32, length(sz)}, fct, sz, pos=pvec)
    return .*(f_sep...)

    # fct = (p, pvec) -> exp((1im*2pi) * pvec * p)
    # return separable_view(fct, sz, pvec)
    # return [exp((1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(sz)]
end

function exp_shift_dat(dat, k_0)
    sz = size(dat)
    # mymid = (sz.÷2).+1
    pvec = 2pi .* k_0 ./ sz;
    # [dat[p] * exp((-1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat)]
    # the collect below, makes it faster
    fct = (p, sz, pvec) -> cis(-pvec * p)
    f_sep = calculate_separables(Array{eltype(dat), length(sz)}, fct, sz, pvec)
    return .*(dat, f_sep...)
    # fct = (p, pvec) -> cis(-pvec * p)
    # sv = separable_view(fct, sz, pvec)
    # # @show size(sv)
    # LazyArray(@~ dat .* sv)
end


function exp_shift_sep(sz, k_0)
    pvec = k_0 ./ sz;
    # return separable_view((p, pvec) -> cis(2pi * pvec * p), sz, pvec)
    fct = (p, sz, pvec) -> cis(2pi * pvec * p)
    f_sep = calculate_separables(Array{ComplexF32, length(sz)}, fct, sz, pos=pvec)
    .*(f_sep...)
end

 # define custom adjoint for sum_exp_shift
 function ChainRulesCore.rrule(::typeof(exp_shift_dat), dat::AbstractArray{T,D}, k_0) where {T,D}
    # collecting is worth it, since the pullback also needs this result
    Z = collect(exp_shift_dat(dat, k_0)) # this yields an array view
    exp_shift_dat_pullback = let sz = size(dat), mymid = (sz.÷2).+1
        function exp_shift_dat_pullback(barx)
            #pvec = 2pi * k_0 ./ sz; # is a cast to Vector helpful?
            #Z2 = dat .* separable_view((p, pvec) -> exp(-1im * pvec * p), sz, pvec)
            # @time res = sum_t(apply_tuple_list.(times_pos, Tuple.(CartesianIndices(sz)), 
            #           barx .* conj.(Z)))
            # @show size(barx) 
            # @time res = sum_t(apply_tuple_list.(times_pos, mypos, barx .* conj.(Z)))
            q = ((Tuple(p) .- mymid) .* (b .* conj.(z))  for (p,b,z) in zip(CartesianIndices(sz), barx, Z))
            res = foldl(.+, q, init=Tuple(zeros(T,D)))
            res = 1im .* res .* 2pi ./ sz
            #@show res
            return NoTangent(), NoTangent(), res # res # apply_tuple_list.(.*, res, barx) # (ChainRulesCore.@not_implemented "Save computation"), 
        end
    end
   # @show abs2(Y)
    return Z, exp_shift_dat_pullback
end


function sum_exp_shift(dat, k_0)
    sz = size(dat)
    # mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    # sum(dat[p] * exp((-1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat))
    fct = (p, sz, pvec) -> cis(-2pi * pvec * p)
    f_sep = calculate_separables(Array{ComplexF32, length(sz)}, fct, sz, pos=pvec)
    return sum(.*(dat, f_sep...))
    # sum(dat .* separable_view((p, pvec) -> exp((-1im*2pi) * pvec * p), sz, pvec))
end

function sum_exp_shift_ix(dat, k_0)
    sz = size(dat)
    mymid = (sz.÷2).+1
    pvec = k_0 ./ sz;
    accumulate(.+, dat[p] .* Tuple(p) .* exp((-1im*2pi) * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat))
end

# """
#     sum_res_i_old(dat::Array{T,D}, pvec::Array{T2,1}) where {T,T2,D}
#     a helper function for the gradient of a sum over an exponential
# """
# function sum_res_i_old(dat::Array{T,D}, pvec::Array{T2,1}) where {T,T2,D}
#     mymid = (size(dat).÷2).+1
#     s1 = sum(conj(dat[p]) * exp(1im * dot(pvec,Tuple(p) .- mymid)) for p in CartesianIndices(dat))
#     s1_i = imag.(s1)
#     s1_r = real.(s1)
#     s2 = zeros(ComplexF64, D)
#     x = zeros(Float64, D)
#     res_i = zeros(Float64, D)
#     s2_r = zeros(Float64, D) 
#     s2_i = zeros(Float64, D) 

#     sp = zero(eltype(pvec))
#     for p in  CartesianIndices(dat)
#         x .= Tuple(p) .- mymid
#         sp = dot(pvec, x)
#         s2 .= x .* (dat[p] .* exp.(-1im .* sp))
#         s2_r .= real.(s2)
#         s2_i .= imag.(s2)
#         res_i .+= s1_i.*s2_r .+ s1_r .* s2_i
#     end
#     res_i
# end


"""
    abs2_ft_peak(dat, k_cur, dims=(1,2))
estimates the complex value of a sub-pixel position defined by `k_cur` in the Fourier-transform of `dat`.
"""
function abs2_ft_peak(dat, k_cur)
    abs2(sum_exp_shift(dat, Tuple(k_cur)))
end

 # define custom adjoint for sum_exp_shift
 function ChainRulesCore.rrule(::typeof(abs2_ft_peak), dat, k_cur)
    # Y = sum_exp_shift(dat, k_cur)
    Z = abs2_ft_peak(dat, k_cur)

    abs2_ft_peak_pullback = let sz = size(dat)
        function abs2_ft_peak_pullback(barx)
            pvec = 2pi .*Vector(k_cur) ./ sz;
            res_i = sum_res_i(dat, pvec)
            # res = 2 .*imag.(s1 .* s2) .* 2pi ./ sz
            res = 2 .* res_i .* 2pi ./ sz
            return NoTangent(), NoTangent(), barx .* res # (ChainRulesCore.@not_implemented "Save computation"), 
        end
    end
   # @show abs2(Y)
    return Z, abs2_ft_peak_pullback
end


