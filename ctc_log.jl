using Zygote: gradient, @ignore
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS

# excitements →  nclasses × ntimeunits
# excitements are already softmax-ed
# labels has blanks inserted at alternate locations

######################################################## Algebra for log-space
logadd() = -Inf
logadd(a::Number) = a
logadd(a::Number, b::Number) = a == -Inf ? b : 
    b == -Inf ? a :
    b > a ? b + log(1 + exp(a - b)) : a + log(1 + exp(b - a))
logadd(a::Number, as...) = logadd(a, logadd(as...))
logadd(as::AbstractArray) = logadd(as...)

function rrule(::typeof(logadd), a, b)          # Needed because of bug in Zygote!!!
    y = logadd(a, b)
    function _logadd_pullback(ȳ)
        p = 1 / (1+ exp(-abs(a-b)))
        isnan(p) && (p = .5)
        ∂a, ∂b  = b < a ? (p, 1-p) : (1-p, p)
        NO_FIELDS, ȳ*∂a, ȳ*∂b
    end
    y, _logadd_pullback
end

################################ Log CTC with Mutation # Needs pullback of logadd
function ctc_log(exite::AbstractMatrix{F}, labels) where F
	blank, T = size(exite)
    U = length(labels)

    lnα = log.(exite[labels, 1]) .+ [0., 0., fill(-Inf, U-2)...]
    lnaddprevprev = @ignore log.([(labels[u] != blank && labels[u] != labels[u-2]) for u in 3:U])

    for t in 2:T
        lnα = vcat(lnα[1], 
                   logadd(lnα[2], lnα[1]), 
                   logadd.(lnα[3:end], lnα[2:end-1], lnaddprevprev.+lnα[1:end-2])) .+ 
              log.(exite[labels, t])
    end
    -logadd(lnα[end-1], lnα[end])
end


############################################# CTC in log scale, Manual gradient
function _ctc_log_loss_and_grad(exite::Array{F}, labels) where F
	blank , T = size(exite)
    U = length(labels)
    addprevprev = [(labels[u] != blank && labels[u] != labels[u-2]) for u in 3:U]

    lnα = Array{F}(undef, U, T)
    lnα[1:2, 1] = log.(exite[labels[1:2], 1])       # Since first label is blank
    lnα[3:end, 1] .= log(zero(F))
    for t in 2:T
        lnα[1, t] = lnα[1, t-1] 
        lnα[2, t] = logadd(lnα[1, t-1], lnα[2, t-1])
        for u in 3:U
            lnα[u, t] = if addprevprev[u-2]
                            logadd(lnα[u-2, t-1], lnα[u-1, t-1], lnα[u, t-1])
                        else
                            logadd(lnα[u-1, t-1], lnα[u, t-1])
                        end
        end
        lnα[:, t] .+= log.(exite[labels, t])
    end

    lnβ = similar(lnα)
    lnβ[1:U-2, T] .= log(zero(F))
    lnβ[U-1:U, T] .= zero(F)
    for t in (T-1):-1:1
        lnβt1 = lnβ[:, t+1] .+ log.(exite[labels, t+1])
        lnβ[U, t] = lnβt1[U]
        lnβ[U-1, t] = logadd(lnβt1[U], lnβt1[U-1])
        for u in 1:U-2
            lnβ[u, t] = if addprevprev[u]
                            logadd(lnβt1[u+2], lnβt1[u+1], lnβt1[u])
                        else
                            logadd(lnβt1[u+1], lnβt1[u])
                        end
        end
    end

    lnαβ = lnα + lnβ
    neglogliklihood = -logadd(lnα[U, T], lnα[U-1, T])
    lnαβadj = lnαβ .+ neglogliklihood

    gradient = zero(exite)
    for u in 1:U
        gradient[labels[u], :] -= exp.(lnαβadj[u, :]) ./ exite[labels[u], :] 
    end

    neglogliklihood, gradient
end

function ctc_log_manual(exite, labels)
    _ctc_log_loss_and_grad(exite, labels)[1]
end

function rrule(::typeof(ctc_log_manual), exite, labels)
    loss, gradients = _ctc_log_loss_and_grad(exite, labels)
    function ctc_pullback(x̄)
        NO_FIELDS, x̄.*gradients, DoesNotExist()
    end
    loss, ctc_pullback
end
