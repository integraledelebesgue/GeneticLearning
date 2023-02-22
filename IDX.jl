module IDX
export read_idx

using Base: ntoh, OneTo
using LinearAlgebra: transpose!


const number_format::Dict{UInt8, DataType} = Dict(
    0x08 => UInt8,
    0x09 => Int8,
    0x0B => Int16,
    0x0C => Int32,
    0x0D => Float32,
    0x0E => Float64
)


function read_idx(file_name)::AbstractArray{Real}

    open(file_name, "r") do file

        magic_number = read(file, 4)

        idx_type::Type = number_format[magic_number[3]]
        dims_count::Int8 = magic_number[4]

        dims = OneTo(dims_count) .|> (_ -> read(file, Int32)) .|> ntoh |> collect |> Tuple
        contents = OneTo(prod(dims)) .|> (_ -> read(file, idx_type)) |> collect

        if idx_type != UInt8 && idx_type != Int8
            contents = ntoh(contents)
        end
        
        reshape(contents, Int64.(dims)...)

    end 

end


end #module
