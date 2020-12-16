include("ctc_plain.jl")
include("ctc_log.jl")
include("ctc_both.jl")
include("ctc_kelley.jl")
using Zygote:gradient
using BenchmarkTools

include("test_head.jl")

### 
# Calculate Loss and Grad robustly
###
manlogloss, manloggrad = _ctc_log_loss_and_grad(exitements, z)
matches(a) = "Invalid"
matches(grad::Matrix) = manloggrad ≈ grad ? "YES" : "NO"
@show manlogloss
@show manloggrad |> extrema
println()

###
# Utilities
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
# Test Loss Calculation
###
@sbt ctc_plain_multiply(exitements, z)
@sbt ctc_plain(exitements, z)
@sbt ctc_plain_both(exitements, z)
@sbt ctc_plain_manual(exitements, z)
# 
@sbt ctc_log(exitements, z)
@sbt ctc_log_both(exitements, z)
@sbt ctc_log_manual(exitements, z)
@sbt ctc_log_manual_kelley(exitements, z)
println()

###
# Define Gradients
###
dctc_plain_multiply(ex, zz) = gradient(ex_ -> ctc_plain_multiply(ex_, zz), ex)[1]
dctc_plain(ex, zz) = gradient(ex_ -> ctc_plain(ex_, zz), ex)[1]
dctc_plain_both(ex, zz) = gradient(ex_ -> ctc_plain_both(ex_, zz), ex)[1]
dctc_plain_manual(ex, zz) = gradient(ex_ -> ctc_plain_manual(ex_, zz), ex)[1]
# 
dctc_log(ex, zz) = gradient(ex_ -> ctc_log(ex_, zz), ex)[1]
dctc_log_both(ex, zz) = gradient(ex_ -> ctc_log_both(ex_, zz), ex)[1]
dctc_log_manual(ex, zz) = gradient(ex_ -> ctc_log_manual(ex_, zz), ex)[1]
dctc_log_manual_kelley(ex, zz) = gradient(ex_ -> ctc_log_manual_kelley(ex_, zz), ex)[1]

###
# Test Gradient Calculation
###
@checkgrad dctc_plain_multiply(exitements, z)
@checkgrad dctc_plain(exitements, z)
@checkgrad dctc_plain_both(exitements, z)
@checkgrad dctc_plain_manual(exitements, z)
# 
@checkgrad dctc_log(exitements, z)
@checkgrad dctc_log_both(exitements, z)          
@checkgrad dctc_log_manual(exitements, z)
@checkgrad dctc_log_manual_kelley(exitements, z)
println()
