using Flux
using Zygote: @adjoint
using Statistics
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS

##
# Code from https://github.com/maetshju/flux-ctc-grad
# Modified to take softmax-ed argument
##
function lgadd(a, b)
  isinf(a) && return b
  isinf(b) && return a
  if a < b
    a, b = b, a
  end
  a + log(1+exp(b-a))
end

function logsum(a::AbstractArray)
  local s
  s = a[1]
  for item in a[2:end]
    s = lgadd(s, item)
  end
  return s
end

function _ctc_kelley(exite, z)
  typedZero = zero(exite[1])
  exite = logsoftmax(exite)
  blank = size(exite, 1)
  T = size(exite, 2)
  U = length(z)

  # Calculate α coefficients, from the upper-left, to the bottom-right
  α = fill(typedZero, T, U)
  for t=1:T
    for u=1:U
      if t == u == 1
        α[t,u] = exite[blank, t]
      elseif t == 1 && u == 2
        α[t,u] = exite[z[2], t]
      elseif t == 1 && u > 2
        α[t,u] = log(typedZero)
      elseif u < U - 2(T - t) - 1
        α[t,u] = log(typedZero)
      else
        idx = u - 2
        idx += z[u] == blank || (u > 2 && z[u-2] == z[u])
        idx = max(1, idx)
        α[t,u] = exite[z[u], t] + logsum(α[t-1, idx:u])
      end
    end
  end

  # Calculate beta coefficients, from the bottom-right, to the upper-left
  β = fill(log(typedZero), T, U)
  β[T,U] = typedZero
  β[T,U-1] = typedZero
  
  # start at T-1 so that β(T, u) = log(0) for all u < U - 1
  for t=(T-1):-1:1
    for u=U:-1:1
      if u > 2t || u > U + 1
        continue
      end
	  
      idx = u+2
      idx -= z[u] == blank || (idx < U && z[u+2] == z[u])
      idx = min(idx, U)

      v = [β[t+1,i] + exite[z[i], t+1] for i=u:idx]
      β[t, u] = logsum(v)
    end
  end
  

  αβ = α + β
  losses = -1 .* mapslices(logsum, αβ, dims=2)

  accum = fill(log(typedZero), size(exite))
  grads = fill(log(typedZero), size(exite))

  for t=1:T
    for u=1:U
      accum[z[u], t] = lgadd(accum[z[u], t], α[t,u] + β[t,u])
    end
    for u=1:size(grads, 1)
      grads[u,t] = exp(exite[u, t]) - exp(accum[u, t] - -losses[t])
    end
  end

  return losses, grads
end

function ctc_log_manual_kelley(exite::Array, y::Array)
  return _ctc_kelley(exite, y)[1] |> mean
end

function rrule(::typeof(ctc_log_manual_kelley), exite, labels)
    losses, gradients = _ctc_kelley(exite, labels)
    _ctc_kelley_pullback(x̄) = (NO_FIELDS, x̄.*gradients, DoesNotExist())
    mean(losses), _ctc_kelley_pullback
end
