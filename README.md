# Explore-CTC-Loss.jl
We explore various ways of implementing the Connectionist Temporal Classification (CTC) Loss function in Julia with Zygote (or not).


```julia
> include("test_ctc.jl")
nclasses = 40
nlabels = 10
ntimesteps = 51
z1 = [17, 3, 15, 33, 19, 8, 31, 25, 1, 19]

manlogloss = 164.68826006918255
manloggrad |> extrema = (-130.3695491682583, 0.0)

ctc_plain_multiply(exitements, z) = 164.68826006918246
  12.166 μs (177 allocations: 42.45 KiB)

ctc_plain(exitements, z) = 164.68826006918246
  87.394 μs (1526 allocations: 137.81 KiB)

ctc_plain_both(exitements, z) = 164.68826006918246
  86.185 μs (1528 allocations: 137.63 KiB)

ctc_plain_manual(exitements, z) = 164.68826006918246
  49.156 μs (884 allocations: 290.16 KiB)

ctc_log(exitements, z) = 164.68826006918255
  132.540 μs (1478 allocations: 125.89 KiB)

ctc_log_both(exitements, z) = 164.68826006918255
  136.563 μs (1528 allocations: 137.63 KiB)

ctc_log_manual(exitements, z) = 164.68826006918255
  127.972 μs (365 allocations: 163.83 KiB)


dctc_plain_multiply(exitements, z) matches?	Yes
  312.818 μs (1801 allocations: 2.05 MiB)

dctc_plain(exitements, z) matches?	Yes
  4.460 ms (22157 allocations: 2.82 MiB)

dctc_plain_both(exitements, z) matches?	Yes
  4.605 ms (22288 allocations: 2.83 MiB)

dctc_plain_manual(exitements, z) matches?	Yes
  50.676 μs (884 allocations: 306.20 KiB)

dctc_log_manual(exitements, z) matches?	Yes
  129.140 μs (365 allocations: 179.88 KiB)

dctc_log(exitements, z) matches?	Yes
  1.282 ms (21175 allocations: 2.55 MiB)

dctc_log_both(exitements, z) errors out!
	DomainError(-1.0, "log will only return a complex result if called with a complex argument. Try log(Complex(x)).")
```