function softmax(x) 
    ex = exp.(x)
    ex ./ sum(ex, dims=1)
end

if false
    nclasses = rand(3:9)
    nlabels = rand(2:7)
    ntimesteps = rand(2nlabels:4nlabels)
    ŷ = repeat(1:ntimesteps, nclasses) ./ 5
    ŷ = reshape(ŷ, nclasses, ntimesteps)
else
    nclasses, nlabels, ntimesteps = 64, 50, 201
    ŷ = randn(nclasses, ntimesteps)
end

z = [nclasses]
for i in 1:nlabels
    append!(z, rand(1:nclasses-1))
    append!(z, nclasses)
end

labels = z[2:2:end]


exitements = softmax(ŷ)
# println("ŷ"), display(ŷ)
# println("softmax(ŷ)"), display(round.(exitements, digits=2))

@show nclasses nlabels ntimesteps labels
println()

