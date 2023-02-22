module Genetics
export crossover, mutate!

using Base: OneTo
using Networks: NeuralNetwork


function random_cut_point(parent1::NeuralNetwork, parent2::NeuralNetwork; parameters::Symbol = :weights)::Int64

    return (parametrs == :weights ? sum.(parent1.shapes) : parent2.layers) |> sum |> OneTo |> rand

end


function crossover(parent1::NeuralNetwork, parent2::NeuralNetwork)::NeuralNetwork

    return NeuralNetwork(
        parent1.layers,
        parent1.shapes,
        parent1.activation,
        unflatten_weights(
            _crossover(parent1.weights, parent2.weights, random_cut_point(parent1, parent2, parameters = :weights)),
            parent1.shapes
        ),
        unflatten_biases(
            _crossover(parent1.biases, parent2.biases, random_cut_point(parent1, parent2, parameters = :biases)),
            parent1.layers    
        )
    )

end


function mutate!(parameters::Vector{VecOrMat{Float64}}, mutation_strength_percent::Float64)::Nothing

    main_index::Int64 = rand(eachindex(parameters))
    parameters[main_index][rand(eachindex(parameters[main_index]))] *= 1.0 + (2rand() - 1.0) * mutation_strength_percent

    return nothing

end


function _crossover(parent1::Vector{VecOrMat{Float64}}, parent2::Vector{VecOrMat{Float64}}, cut_point::Integer)::Vector{Float64}

    return vcat(
        Iterators.take(Iterators.flatten(parent1), cut_point),
        Iterators.drop(Iterators.flatten(parent2), cut_point)
    ) |> Iterators.flatten |> collect

end


function unflatten_weights(numbers::Vector{Float64}, shapes::Vector{Tuple{Int64, Int64}})::Vector{Matrix{Float64}}

    lengths::Vector{Int64} = accumulate(+, prod.(shapes))
    
    ranges::Vector{UnitRange{Int64}} = zip(
        [1; lengths[1:end-1] .+ 1],
        lengths
    ) .|> (rng::Tuple{Int64, Int64} -> UnitRange(rng...))
    
    matrix_views::Vector{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}} = 
        ranges .|> (rng::UnitRange{Int64} -> view(numbers, rng))

    return zip(
        matrix_views,
        shapes
    ) .|> ((mtrx_view, shape)::Tuple -> reshape(mtrx_view, shape)) .|> collect

end


function unflatten_biases(numbers::Vector{Float64}, layers::Vector{Int64})::Vector{Vector{Float64}}

    lengths::Vector{Int64} = accumulate(+, layers)

    ranges::Vector{UnitRange{Int64}} = zip(
        [1; lengths[1:end-1] .+ 1],
        lengths
    ) .|> (rng::Tuple{Int64, Int64} -> UnitRange(rng...))

    return ranges .|> (rng::UnitRange{Int64} -> view(numbers, rng)) .|> collect

end


end #module 