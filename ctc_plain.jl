using LinearAlgebra
using Zygote: @ignore
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS

# excitements → nclasses × ntimeunits
# excitements are already softmax-ed
# labels has blanks inserted at alternate locations

################################ Plain CTC with Multiplication 

function recurrence(labels, blank)  # Transpose is what is used
    U = length(labels)
    A = Matrix{Float64}(I, U, U)            #        α[u, t] += α[u, t-1] 
    A[1, 2] = 1.
    for u in 3:U
        A[u-1, u] = 1.                     #         α[u, t] += α[u-1, t-1]
        A[u-2, u] = (labels[u] != blank && labels[u] != labels[u-2])
    end
    A
end

function ctc_plain_multiply(exite::AbstractMatrix{F}, labels) where F
    blank, T = size(exite)
    exite = softmax(exite)
    α = exite[labels, 1] .* vcat(one(F), one(F), zeros(F, length(labels)-2))
    R = @ignore recurrence(labels, blank)

    for t in 2:T
        α = R'*α .* exite[labels, t]
    end
    -log(α[end] + α[end-1])
end


################################ Plain CTC 
function ctc_plain(exite::AbstractMatrix{F}, labels) where F
	blank, T = size(exite)
    U = length(labels)
    exite = softmax(exite)
    α = exite[labels, 1] .* vcat(one(F), one(F), zeros(F, U-2))
    addprevprev = @ignore [(labels[u] != blank && labels[u] != labels[u-2]) for u in 3:U]

    for t in 2:T
        α = vcat(α[1], α[2]+α[1], α[3:U]+α[2:U-1]+addprevprev.*α[1:U-2]) .* exite[labels, t]
    end
    -log(α[U] + α[U-1])
end

################################ Plain CTC - Manual gradient
function _ctc_plain_loss_and_grad(exite::AbstractMatrix{F}, labels) where F
	blank, T = size(exite)
    U = length(labels)
    addprevprev = [(labels[u] != blank && labels[u] != labels[u-2]) for u in 3:U]

    exite = softmax(exite)
    α = Array{F}(undef, U, T)
    α[1:2, 1] = exite[labels[1:2], 1]       # Since first label is blank
    α[3:end, 1] .= zero(F)

    for t in 2:T
        α[1, t] = α[1, t-1]
        α[2, t] = α[1, t-1] + α[2, t-1]
        α[3:U, t] = α[3:U, t-1] + α[2:U-1, t-1] + addprevprev .* α[1:U-2, t-1]
        α[:, t] .*= exite[labels, t]
    end

    β = similar(α)
    β[1:U-2, T] .= zero(F)
    β[U-1:U, T] .= one(F)
    for t in T-1:-1:1
        βt1 = β[:, t+1] .* exite[labels, t+1]
        β[U, t] = βt1[U]
        β[U-1, t] = βt1[U] + βt1[U-1]
        β[1:U-2, t] = βt1[1:U-2] + βt1[2:U-1] + addprevprev .* βt1[3:U]
    end

    liklihood = α[U, T] + α[U-1, T]
    αβl = α .* β ./ liklihood
    gradient = exite
    for u in 1:U
        gradient[labels[u], :] -= αβl[u, :] 
    end
    -log(liklihood), gradient
end

function ctc_plain_manual(exite, labels)
    _ctc_plain_loss_and_grad(exite, labels)[1]
end

function rrule(::typeof(ctc_plain_manual), exite, labels)
    loss, gradients = _ctc_plain_loss_and_grad(exite, labels)
    ctc_manual_pullback(x̄) = (NO_FIELDS, x̄.*gradients, DoesNotExist())
    loss, ctc_manual_pullback
end
