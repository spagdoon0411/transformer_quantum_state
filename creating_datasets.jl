using Pkg
using SparseArrays
using Profile
using PProf
using ArnoldiMethod

⊗(x, y) = kron(x, y)

id = [1 0; 0 1] |> sparse
σˣ = [0 1; 1 0] |> sparse
σᶻ = [1 0; 0 -1] |> sparse

l = ReentrantLock()

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

precompile(TransverseFieldIsing_sparse_threaded, (Int, Float64))
precompile(ground_state, (Int, Float64))
precompile(generate_ground, (Int, Float64))

# Run once to force compilation, for accurate profiling
# TransverseFieldIsing_sparse(N=20, h=1.0)

# Profile.clear()
# @profile TransverseFieldIsing_sparse(N=20, h=1.0)
# Profile.print()
# pprof()

for h in 0.1:0.1:4.0
    @time generate_ground(N=18, h=h)
end