###
# Display and Timing Utilities
###
disp3(a) = display(round.(a, digits=1))
macro sbt(ex)
    quote 
        @show $ex
        @btime $ex
        println() 
    end
end

macro checkgrad(ex)
    strex = string(ex)
    quote
        try 
            println($strex, " matches? ", $matches($ex))
            @btime $ex 
            println()
        catch err
            println($strex, " errors out!\n\t", err, "\n")
        end
       nothing
    end
end

###
# Softmax
###

function softmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    sum_exp = sum(exp_, dims=dims)
    exp_ ./ sum_exp
end

function logsoftmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    sum_exp = sum(exp_, dims=dims)
    log_ = log.(sum_exp)
    (xs .- max_) .- log_
end

function softmaxandlog(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    sum_exp_ = sum(exp_, dims=dims)
    softmax_ = exp_ ./ sum_exp_ 
    logsoftmax_ = @. xs - max_ - log(sum_exp_)
    softmax_, logsoftmax_
end