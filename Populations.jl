module Populations
export Population

using Networks: NeuralNetwork
using Genetics: crossover, mutate!


struct Population

    agents::Vector{NeuralNetwork}

    weight_mutation_chance::Float64
    bias_mutation_chance::Float64

    weight_mutation_strength_percent::Float64
    bias_mutation_strength_percent::Float64

    training_function::Function
    
    top_count::Int64

    
    Population(
        agents_count::Int64, 
        network_shape::Vector{Int64}, 
        activation_function::Function,
        weight_mutation_chance::Float64,
        bias_mutation_chance::Float64,
        weight_mutation_strength_percent::Float64,
        bias_mutation_strength_percent::Float64, 
        training_function::Function,
        top_percent::Float64
    )::Population = new(
        [NeuralNetwork(network_shape, activation_function) for _ in 1:agents_count],
        weight_mutation_chance::Float64,
        bias_mutation_chance::Float64,
        weight_mutation_strength_percent::Float64,
        bias_mutation_strength_percent::Float64,
        training_function,
        floor(Int64, top_percent * agents_count)
    )

end


AgentsView::Type = SubArray{NeuralNetwork, 1, Vector{NeuralNetwork}, Tuple{UnitRange{Int64}}, true}

function next_generation!(population::Population, inputs::Vector{Vector{Float64}}, outputs::Vector{Vector{Float64}})::Nothing

    for agent::NeuralNetwork in population.agents
        agent.fitness[] = population.training_function(agent, inputs, outputs)
    end

    sort!(
        population.agents, 
        by = agent::NeuralNetwork -> agent.fitness
    )

    best::AgentsView = @view(population.agents[1:population.top_count])
    to_replace::AgentsView = @view(population.agents[population.top_count+1 : end])

    map!(
        agent::NeuralNetwork -> crossover(rand(best, 2)...),
        to_replace,
        to_replace
    )

    for agent::NeuralNetwork in population.agents
        if rand() < population.weight_mutation_chance
           mutate!(agent.weights, population.weight_mutation_strength_percent) 
        end

        if rand() < population.bias_mutation_chance
            mutate!(agent.biases, population.bias_mutation_strength_percent)
        end
    end

end


function evolve(population::Population, max_generations::Int64, convergence_threshold::Float64, inputs::Vector{Vector{Float64}}, outputs::Vector{Vector{Float64}})::NeuralNetwork

    return let best_agent::NeuralNetwork

        for generation::Int64 in 1:max_generations

            next_generation!(population, inputs, outputs)

            best_agent = population.agents[1]

            if best_agent.fitness[] <= convergence_threshold
                break
            end

        end

        best_agent

    end

end


end #module