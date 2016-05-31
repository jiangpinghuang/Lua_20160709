--[[
Some examples and functions are studied in this file.
--]]

require 'nn'

-- LookupTable --
--[[
module = nn.LookupTable(nIndex, size, [paddingValue], [maxNorm], [normType])

This layer is a particular case of a convolution, where the width of the convolution would be 1. When calling forward(input), 
it assumes input is a 1D or 2D tensor filled with indices. If the input is a matrix, then each row is assumed to be an input 
sample of given batch. Indices start at 1 and can go up to nIndex. For each index, it outputs a corresponding Tensor of size 
specified by size.

LookupTable can be very slow if a certain input occurs frequently compared to other inputs; this is often the case for input 
padding. During the backward step, there is a separate thread for each input symbol which results in a bottleneck for frequent 
inputs. generating a n x size1 x size2 x ... x sizeN tensor, where n is the size of a 1D input tensor.

Again with a 1D input, when only size1 is provided, the forward(input) is equivalent to performing the following matrix-matrix 
multiplication in an efficient manner:

M P

where M is a 2D matrix size x nIndex containing the parameters of the lookup-table and P is a 2D matrix, where each column vector 
i is a zero vector except at index input[i] where it is 1.
--]]

-- 1D example --
module = nn.LookupTable(10, 3)

input = torch.Tensor{1, 2, 1, 2}
print(input)

-- 2D example -- 
module = nn.LookupTable(10, 3)
input = torch.Tensor({1, 2, 4, 5}, {4, 3, 2, 10})
print(module:forward(input))
print(module:forward(input))
