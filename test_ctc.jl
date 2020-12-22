include("utils.jl")
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
manlogloss, manloggrad = _ctc_plain_loss_and_grad(exitements, labels)
matches(a) = "Invalid"
matches(grad::Matrix) = manloggrad ≈ grad ? "YES" : "NO"
@show manlogloss
@show manloggrad |> extrema
println()

###
# Test Loss Calculation
###
@sbt ctc_plain_multiply(exitements, labels)
@sbt ctc_plain(exitements, labels)
@sbt ctc_plain_both(exitements, labels)
@sbt ctc_plain_manual(exitements, labels)
# 
@sbt ctc_log(exitements, labels)
@sbt ctc_log_both(exitements, labels)
@sbt ctc_log_manual(exitements, labels)
@sbt ctc_log_manual_kelley(exitements, labels)
println()

###
# Define Gradients
###
dctc_plain_multiply(ex, lbl) = gradient(ex_ -> ctc_plain_multiply(ex_, lbl), ex)[1]
dctc_plain(ex, lbl) = gradient(ex_ -> ctc_plain(ex_, lbl), ex)[1]
dctc_plain_both(ex, lbl) = gradient(ex_ -> ctc_plain_both(ex_, lbl), ex)[1]
dctc_plain_manual(ex, lbl) = gradient(ex_ -> ctc_plain_manual(ex_, lbl), ex)[1]
# 
dctc_log(ex, lbl) = gradient(ex_ -> ctc_log(ex_, lbl), ex)[1]
dctc_log_both(ex, lbl) = gradient(ex_ -> ctc_log_both(ex_, lbl), ex)[1]
dctc_log_manual(ex, lbl) = gradient(ex_ -> ctc_log_manual(ex_, lbl), ex)[1]
dctc_log_manual_kelley(ex, lbl) = gradient(ex_ -> ctc_log_manual_kelley(ex_, lbl), ex)[1]

###
# Test Gradient Calculation
###
@checkgrad dctc_plain_multiply(exitements, labels)
@checkgrad dctc_plain(exitements, labels)
@checkgrad dctc_plain_both(exitements, labels)
@checkgrad dctc_plain_manual(exitements, labels)
# 
@checkgrad dctc_log(exitements, labels)
@checkgrad dctc_log_both(exitements, labels)          
@checkgrad dctc_log_manual(exitements, labels)
@checkgrad dctc_log_manual_kelley(exitements, labels)
println()
