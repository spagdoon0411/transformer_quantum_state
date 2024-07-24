using Pkg
using SparseArrays
using Profile
using PProf
using ArnoldiMethod

⊗(x, y) = kron(x, y)

id = [1.0 0.0; 0.0 1.0] |> sparse
σˣ = [0.0 1.0; 1.0 0.0] |> sparse
σᶻ = [1.0 0.0; 0.0 -1.0] |> sparse

l = ReentrantLock()

# Produces the first two terms of the TFIM Hamiltonian without 
# coefficients applied (i.e., for h = 1.0)
function TransverseFieldIsing_sparse_threaded_base(; N::Int64)

    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ

    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ

    H1 = spzeros(Int, 2^N, 2^N) # note the spzeros instead of zeros here
    H2 = spzeros(Int, 2^N, 2^N)

    Threads.@threads for i in 1:N
        this_thread_first_term_ops = circshift(first_term_ops, i - 1)
        term = foldl(⊗, this_thread_first_term_ops)
        lock(l)
        try
            H1 -= term # TODO: optimized via OpenBLAS?
        finally
            unlock(l)
        end
    end

    Threads.@threads for i in 1:N
        this_thread_second_term_ops = circshift(second_term_ops, i - 1)
        term = foldl(⊗, this_thread_second_term_ops)
        lock(l)
        try
            H2 -= term
        finally
            unlock(l)
        end
    end

    H1, H2
end

# "Specializes" a base TFIM Hamiltonian to a specific value of h
# by performing a linear combination of the two terms
# TODO: does type conversion from Int64 to Float64 have overhead?
@inline 
function specialize_H_threaded(H1::SparseMatrixCSC{Float64}, H2::SparseMatrixCSC{Float64}, h::Float64)
    H1 + h * H2 # TODO: is this operation optimized by Julia via OpenBLAS?
end

function TransverseFieldIsing_sparse(; N, h)

    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ

    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ

    H = spzeros(Int, 2^N, 2^N) # note the spzeros instead of zeros here

    for i in 1:N
        H -= foldl(⊗, first_term_ops)
        first_term_ops = circshift(first_term_ops, 1)
    end

    for i in 1:N
        H -= h * foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops, 1)
    end

    H
end

function TransverseFieldIsing_sparse_threaded(; N, h)

    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ

    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ

    H = spzeros(Int, 2^N, 2^N) # note the spzeros instead of zeros here

    Threads.@threads for i in 1:N
        this_thread_first_term_ops = circshift(first_term_ops, i - 1)
        term = foldl(⊗, this_thread_first_term_ops)
        lock(l)
        try
            H -= term
        finally
            unlock(l)
        end
    end

    Threads.@threads for i in 1:N
        this_thread_second_term_ops = circshift(second_term_ops, i - 1)
        term = foldl(⊗, this_thread_second_term_ops)
        lock(l)
        try
            H -= h * term
        finally
            unlock(l)
        end
    end

    H
end


function ground_state(H)
    decomposition, history = partialschur(H, nev=1, which=:SR)
    energies, states = partialeigen(decomposition)
    return energies[1], states[:, 1]
end

function generate_ground(; N, h)
    H = TransverseFieldIsing_sparse(N=N, h=h)
    ground_state(H)
end

# H = TransverseFieldIsing_sparse(N=20, h=1.0)
# H_threaded = TransverseFieldIsing_sparse_threaded(N=20, h=1.0)
# 
# diff = H - H_threaded
# diff_sum = sum(diff)
# print(diff_sum) # How close to zero is this?

# precompile(TransverseFieldIsing_sparse_threaded, (Int, Float64))
# precompile(ground_state, (Int, Float64))
# precompile(generate_ground, (Int, Float64))

# Run once to force compilation, for accurate profiling
# TransverseFieldIsing_sparse(N=20, h=1.0)

# Profile.clear()
# @profile TransverseFieldIsing_sparse(N=20, h=1.0)
# Profile.print()
# pprof()

precompile(TransverseFieldIsing_sparse_threaded_base, (Int,))
precompile(specialize_H_threaded, (SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{Float64, Int64}, Float64))

Threads.@threads for N in 4:1:20
    @time H1, H2 = TransverseFieldIsing_sparse_threaded_base(N=N)
    for h in 0.1:0.1:4.0
        @time ground_state(specialize_H_threaded(H1, H2, h))
    end
end