# Explore-CTC-Loss.jl
We explore various ways of implementing the Connectionist Temporal Classification (CTC) Loss function in Julia with Zygote (or not). We also compare it with the existing [https://github.com/maetshju/flux-ctc-grad].


```julia
> include("test_ctc.jl")
alphabetsize = 64
textlength = 50
ntimesteps = 201
rawlabels = [57, 41, ..., 59]

manlogloss = 705.237164456796
manloggrad |> extrema = (-0.84280330876603, 0.41492393816423795)

ctc_plain_multiply(exitements, labels) = 705.237164456796
  1.238 ms (636 allocations: 813.91 KiB)

ctc_plain(exitements, labels) = 705.237164456796
  689.125 μs (6035 allocations: 1.73 MiB)

ctc_plain_both(exitements, labels) = 705.237164456796
  692.193 μs (6036 allocations: 1.73 MiB)

ctc_plain_manual(exitements, labels) = 705.237164456796
  912.961 μs (3322 allocations: 3.75 MiB)

ctc_log(exitements, labels) = 705.237164456796
  1.562 ms (5918 allocations: 1.56 MiB)

ctc_log_both(exitements, labels) = 705.2371644567957
  1.704 ms (6036 allocations: 1.73 MiB)

ctc_log_manual(exitements, labels) = 705.2371644567957
  2.465 ms (1427 allocations: 2.47 MiB)

ctc_log_manual_kelley(exitements, labels) = 705.2371644567961
  28.302 ms (601989 allocations: 19.08 MiB)


dctc_plain_multiply(exitements, labels) matches? YES
  21.584 ms (45848 allocations: 73.82 MiB)

dctc_plain(exitements, labels) matches? YES
  25.369 ms (126970 allocations: 49.47 MiB)

dctc_plain_both(exitements, labels) matches? YES
  25.246 ms (127017 allocations: 49.47 MiB)

dctc_plain_manual(exitements, labels) matches? YES
  925.725 μs (3323 allocations: 3.85 MiB)

dctc_log(exitements, labels) matches? YES
  12.320 ms (217141 allocations: 51.70 MiB)

dctc_log_both(exitements, labels) errors out!
	DomainError(-1.0, "Try log(Complex(x)).")

dctc_log_manual(exitements, labels) matches? YES
  2.432 ms (1428 allocations: 2.57 MiB)

dctc_log_manual_kelley(exitements, labels) matches? YES
  28.087 ms (602018 allocations: 19.18 MiB)

```