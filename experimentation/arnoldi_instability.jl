using SparseArrays
using ArnoldiMethod
using LinearAlgebra

N0 = -5
Nf = 5
dN = 1
A = spdiagm(0 => N0:dN:Nf)
off_diag_perturb = 1e-5
A += off_diag_perturb * sprandn(11, 11, 0.5)

decomp1, hist1 = partialschur(A, nev=1, which=:SR)
energies1, states1 = partialeigen(decomp1)

decomp2, hist2 = partialschur(A, nev=1, which=:SR)
energies2, states2 = partialeigen(decomp2)

state1 = states1[:, 1]
state2 = states2[:, 1]

normdiff = norm(state1 - state2)
println("Norm of state difference: $normdiff")  

@assert state1 ≈ state2