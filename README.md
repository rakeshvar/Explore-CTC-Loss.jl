# Explore-CTC-Loss.jl
We explore various ways of implementing the Connectionist Temporal Classification (CTC) Loss function in Julia with Zygote (or not). We also compare it with the existing [https://github.com/maetshju/flux-ctc-grad].


```julia
> include("test_ctc.jl")
nclasses = 64
nlabels = 50
ntimesteps = 201
labels = [20, 9, ..., 57]

manlogloss = 689.8988368628794
manloggrad |> extrema = (-191.85841465326217, 0.0)

ctc_plain_multiply(exitements, z) = 689.8988368628798
  789.006 μs (708 allocations: 609.92 KiB)

ctc_plain(exitements, z) = 689.8988368628798
  462.985 μs (6026 allocations: 1.53 MiB)

ctc_plain_both(exitements, z) = 689.8988368628798
  464.565 μs (6108 allocations: 1.53 MiB)

ctc_plain_manual(exitements, z) = 689.8988368628798
  800.852 μs (3618 allocations: 4.12 MiB)

ctc_log(exitements, z) = 689.8988368628794
  1.467 ms (5908 allocations: 1.36 MiB)

ctc_log_both(exitements, z) = 689.8988368628794
  1.453 ms (6108 allocations: 1.53 MiB)

ctc_log_manual(exitements, z) = 689.8988368628794
  2.425 ms (1520 allocations: 2.44 MiB)

ctc_log_manual_kelley(exitements, z) = 689.8988368628796
  27.119 ms (613761 allocations: 19.14 MiB)


dctc_plain_multiply(exitements, z) matches? YES
  16.697 ms (7873 allocations: 71.77 MiB)

dctc_plain(exitements, z) matches? YES
  24.078 ms (88304 allocations: 47.39 MiB)

dctc_plain_both(exitements, z) matches? YES
  24.381 ms (89038 allocations: 47.42 MiB)

dctc_plain_manual(exitements, z) matches? YES
  808.404 μs (3619 allocations: 4.21 MiB)

dctc_log(exitements, z) matches? YES
  13.325 ms (244773 allocations: 51.87 MiB)

dctc_log_both(exitements, z) errors out!
	DomainError(-1.0, "Try log(Complex(x)).")

dctc_log_manual(exitements, z) matches? YES
  2.441 ms (1521 allocations: 2.54 MiB)

dctc_log_manual_kelley(exitements, z) matches? YES
  29.229 ms (613790 allocations: 19.24 MiB)

```