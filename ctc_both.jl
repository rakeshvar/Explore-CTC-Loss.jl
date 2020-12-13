###############################
# Log domain arthmetic 
###############################
import Base: +, *, ==, log, /

struct LogNum{F<:AbstractFloat} <: AbstractFloat
    v::F
    LogNum(loga::F, ::Val{:log}) where F = new{F}(loga)
    LogNum{F}(a) where F = new{F}(log(a))
end

LogNum(a::F) where F = LogNum{F}(a)
*(a::LogNum, b::LogNum) = LogNum(a.v+b.v, Val(:log))
function +(a::LogNum, b::LogNum)
    a.v == -Inf  &&         return b
    b.v == -Inf  &&         return a
    v = if a.v > b.v
        a.v + log(1 + exp(b.v-a.v))
    else 
        b.v + log(1 + exp(a.v-b.v))
    end
    LogNum(v, Val(:log))
end
*(a::LogNum, v::AbstractFloat) = LogNum(a.v+log(v), Val(:log))
*(b::Bool, a::LogNum{F}) where F = b ? a : LogNum(zero(F))
log(a::LogNum) = a.v

/(a::LogNum, b::LogNum) = LogNum(a.v-b.v, Val(:log))

###############################
# CTC - Works in Both Domains 
###############################
function _ctc_both(exite::AbstractMatrix{F}, labels, ::Val{uselog}) where {F<:AbstractFloat, uselog}
	blank, T = size(exite)
    U = length(labels)
    L = uselog ? LogNum{F} : F
    α = Array{L}(exite[labels, 1] .* [1., 1., zeros(U-2)...])
    addprevprev = @ignore [(labels[u] != blank && labels[u] != labels[u-2]) for u in 3:U]

    for t in 2:T
        α = vcat(α[1], α[2]+α[1], α[3:U]+α[2:U-1]+addprevprev.*α[1:U-2]) .* exite[labels, t]
    end
    -log(α[U] + α[U-1])
end

ctc_plain_both(exite, labels) = _ctc_both(exite, labels, Val(false))
ctc_log_both(exite, labels) = _ctc_both(exite, labels, Val(true))
