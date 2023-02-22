push!(LOAD_PATH, @__DIR__)

using Networks: NeuralNetwork, output
using Populations: Population, evolve
using IDX: read_idx
using Colors, Plots

logistic::Function = x::Float64 -> 1 / (1 + exp(-x))


function test_output()

    println("First network: ")
    nn = NeuralNetwork([10, 20, 48, 15, 5], logistic)

    for _ in 1:10
        @time output(nn, rand(10))
    end

    println("\nSecond network: ")

    nn2 = NeuralNetwork([5, 12, 22, 50, 13, 8], tanh)

    for _ in 1:10
        @time output(nn2, rand(5))
    end

end


function main()::Nothing

    population::Population = Population(
        10,
        [1, 3, 5, 3, 1],
        logistic,
        0.05,
        0.05,
        0.1,
        0.1,
        identity,
        0.1
    )

    data = read_idx("$(@__DIR__)/Datasets/train-images.idx3-ubyte")
    labels = read_idx("$(@__DIR__)/Datasets/train-labels.idx1-ubyte")

    item_number::Int64 = rand(1:10000)

    image::Matrix{Float64} = data[item_number, :, :] ./ 255.0

    #(Float64.(data[item_number, :, :]) ./ 255.0) .|> Colors.Gray |> plot |> display
    heatmap(
        image,
        color = :greys,
        aspectratio = :equal
    ) |> display

    println(Int8(labels[item_number]))

    return nothing

end


main()
