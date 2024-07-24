using SparseArrays, ArnoldiMethod, Arrow, JSON, DataFrames, Dates

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
        this_thread_first_term_ops = circshift(first_term_ops, i)
        term = foldl(⊗, this_thread_first_term_ops)
        lock(l)
        try
            H1 -= term # TODO: optimized via OpenBLAS?
        finally
            unlock(l)
        end
    end

    Threads.@threads for i in 1:N
        this_thread_second_term_ops = circshift(second_term_ops, i)
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
function specialize_H_threaded(H1::SparseMatrixCSC{Float64,Int64}, H2::SparseMatrixCSC{Float64,Int64}, h::Float64)
    H1 + h * H2 # TODO: is this operation optimized by Julia via OpenBLAS?
end

# # "Specializes" a base TFIM Hamiltonian to a specific value of h
# # by performing a linear combination of the two terms
# # TODO: does type conversion from Int64 to Float64 have overhead?
# @inline
# function specialize_H_threaded(H1::SparseMatrixCSC{Int64,Int64}, H2::SparseMatrixCSC{Int64,Int64}, h::Float64)
#     H1 + h * H2 # TODO: is this operation optimized by Julia via OpenBLAS?
# end

function ground_state(H)
    decomposition, history = partialschur(H, nev=1, which=:SR)
    energies, states = partialeigen(decomposition)
    return energies[1], states[:, 1]
end

h_min = 0.5
h_max = 1.5
h_step = 0.1
N_min = 2
N_max = 10
N_step = 1
Nrange = N_min:N_step:N_max
hrange = h_min:h_step:h_max

top_dataset_dir = "TFIM_ground_states"

timestamp = replace(string(Dates.now()), ":" => "-")
this_dataset_dir = "$timestamp"

table_dir = joinpath(top_dataset_dir, this_dataset_dir)
mkpath(table_dir)

file_names = fill("", length(Nrange))

N_index = 1
for N in Nrange # TODO: allocate threads more explicitly
    file_name = joinpath(table_dir, "$N.arrow")
    file_names[N_index] = file_name
    table_file = open(file_name, "w")

    ground_state_df = DataFrame(N=Int[], h=Float64[], energy=Float64[], state=Vector{Float64}[])

    println("Switching to N = $N")
    @time H1, H2 = TransverseFieldIsing_sparse_threaded_base(N=N)
    for h in hrange
        println("N = $N, h = $h")
        @time energy, state = ground_state(specialize_H_threaded(H1, H2, h))
        push!(ground_state_df, (N, h, energy, abs.(state)))
    end

    Arrow.write(table_file, ground_state_df, file=true)
    close(table_file)
    global N_index += 1
end

meta_path = joinpath(table_dir, "meta.json")
meta_file = open(meta_path, "w")
meta = Dict(
    "N_min" => N_min,
    "N_max" => N_max,
    "N_step" => N_step,
    "h_min" => h_min,
    "h_max" => h_max,
    "h_step" => h_step,
    "file_names" => file_names
)
JSON.print(meta_file, meta)
