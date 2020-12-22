
if false
    alphabetsize = rand(3:9)
    textlength = rand(2:7)
    ntimesteps = rand(2textlength:4textlength)
    exitements = repeat(1:ntimesteps, alphabetsize) ./ 5
    exitements = reshape(exitements, alphabetsize, ntimesteps)
else
    alphabetsize, textlength, ntimesteps = 64, 50, 201
    exitements = randn(alphabetsize, ntimesteps)
end

blank = alphabetsize
labels = [blank]
for i in 1:textlength
    append!(labels, rand(1:blank-1))
    append!(labels, blank)
end

rawlabels = labels[2:2:end]

# println("exitements")
# display(round.(exitements, digits=2))

@show alphabetsize textlength ntimesteps rawlabels
println()
