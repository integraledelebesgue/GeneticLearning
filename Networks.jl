module Networks
export NeuralNetwork, output

# Simplified, barely optimized implementation

struct NeuralNetwork

    layers::Vector{Int64}
    shapes::Vector{Tuple{Int64, Int64}}

    activation::Function

    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}

    fitness::Ref{Float64}

    NeuralNetwork(
        layers::Vector{Int64}, 
        activation_function::Function
    )::NeuralNetwork = 
        new(_NeuralNetwork(layers, activation_function)...)
    
    NeuralNetwork(
        layers::Vector{Int64}, 
        shapes::Vector{Tuple{Int64, Int64}}, 
        activation::Function, 
        weights::Vector{Matrix{Float64}}, 
        biases::Vector{Matrix{Float64}}
    )::NeuralNetwork = 
        new(layers, shapes, activation, weights, biases, Ref(0.0))

end        


ConstructorReturn::Type = Tuple{Vector{Int64}, Vector{Tuple{Int64, Int64}}, Function, Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Ref{Float64}}

function _NeuralNetwork(layers::Vector{Int64}, activation_function::Function)::ConstructorReturn
    
    _shapes::Vector{Tuple{Int64, Int64}} = shapes(layers)
    
    _weights::Vector{Matrix{Float64}} = 
        _shapes .|> (shape::Tuple{Int64, Int64} -> randn(shape...))

    _biases::Vector{Vector{Float64}} = layers[2:end] .|> randn

    return (
        layers,
        _shapes,
        activation_function,
        _weights,
        _biases,
        Ref(0.0)    
    )

end


function shapes(layers::Vector{Int64})::Vector{Tuple{Int64, Int64}}

    return eachindex(layers[1:end-1]) .|> (i -> (layers[i+1], layers[i]))

end 


function output(network::NeuralNetwork, input::Vector{Float64})::Vector{Float64}

    layer_output::Vector{Float64} = input

    for (layer_weights::Matrix{Float64}, layer_biases::Vector{Float64}) in zip(@view(network.weights[:]), @view(network.biases[:]))
        layer_output = network.activation.(layer_weights * layer_output + layer_biases)
    end

    return layer_output
    
end


end #module