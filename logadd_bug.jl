using Zygote

############ Recurrence relation for state of the system
A = [1 0 0; 
     1 1 0; 
     1 1 1]

################## Loss of the dynamical system
function loss(x)                               # Input x > 0
    state = x[:, 1]                            # state_0 = [1, 0, 0, ...] => state_1 = x_1
    for i in 2:size(x)[2]
        state = A*state                        # Update state depending on previous state and
        state = state .* x[:, i]               # Current input
    end
    state[end]                                 # loss is a simple function of state
end

#################################  Generate some simple data
X = reshape((1:3*7).%5, 3, 7) .+ 1
display(X)
@show loss(X)
display(gradient(loss, X)[1])

################################# Do the same operation in the log-domain
logadd1(a, b) = log(exp(a)+exp(b))
logadd2(a, b) = b + log(1 + exp(a - b))
logadd3(a, b) = b > a ? b + log(1 + exp(a - b)) : a + log(1 + exp(b - a))
logadd4(a, b) = a == -Inf ? b : 
                b == -Inf ? a :
                b > a ? b + log(1 + exp(a - b)) : a + log(1 + exp(b - a))

logadd = logadd3 # Pick one of the four implementations

function logloss(x)
    logstate = log.(x[:, 1])
    for i in 2:size(x)[2]
        logstate = [logstate[1], 
                    logadd(logstate[1], logstate[2]), 
                    logadd(logstate[1], logadd(logstate[2], logstate[3]))]
        logstate += log.(x[:, i])
    end
    exp(logstate[end])
end

@show logloss(X)
display(gradient(logloss, X)[1])

for i in 1:10
    a, b = randn(2)
    println((logadd1(a, b), logadd2(a, b), logadd3(a, b), logadd4(a, b)))
    println((gradient(logadd1, a, b), gradient(logadd2, a, b), gradient(logadd3, a, b), gradient(logadd4, a, b)))
end