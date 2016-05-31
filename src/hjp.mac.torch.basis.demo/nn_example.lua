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

input = torch.Tensor{1, 2, 1, 10}
print(input)
print(module:forward(input))

-- 2D example -- 
mod = nn.LookupTable(10, 3)
inputs = torch.Tensor({1, 2, 4, 5}, {4, 3, 2, 10})
print(mod:forward(inputs))

-- max-norm regularization --
module = nn.LookupTable(10, 3, 0, 1, 2)
input = torch.Tensor{1, 2, 1, 10}
print(module.weight)
print(module:forward(input))
print(module.weight)


-- Reshape --
x = torch.Tensor(4, 4)
print(x)
for i = 1, 4 do
  for j = 1, 4 do
    x[i][j] = (i-1)*4 + j
  end
end
print(x)

print(nn.Reshape(2, 8):forward(x))
print(nn.Reshape(8, 2):forward(x))
print(nn.Reshape(16, 1):forward(x))
print(nn.Reshape(1, 16):forward(x))
print(nn.Reshape(4, 4):forward(x))
print(nn.Reshape(16):forward(x))

y = torch.Tensor(1, 4):fill(3)
print(y)
print(nn.Reshape(4, 1):forward(y))
print(nn.Reshape(4, false):forward(y))
print(nn.Reshape(4, true):forward(y))

-- cAddTable()() --
ii = {torch.ones(5), torch.ones(5)*2, torch.ones(5)*3}
print(ii[1])
print(ii[2])
print(ii[3])
m = nn.CAddTable()
print(m:forward(ii))

-- Max --
t = torch.randn(4, 5, 6)
print(t)
module = nn.Max(2)
print(module:forward(t))

-- Padding --
module = nn.Padding(1, 2, 1, -1)
t = torch.randn(3)
print(t)
print(module:forward(t))

module = nn.Padding(1, -2, 1, -1)
t = torch.randn(2, 3)
print(t)
print(module:forward(t))

module = nn.Padding(1,-2, 1, -1, 2)
t = torch.randn(2, 3)
print(t)
print(module:forward(t))

-- TemporalConvolution --
inp = 5; outp = 1; kw = 1; dw = 1;
mlp = nn.TemporalConvolution(inp, outp, kw, dw)
x = torch.rand(7, inp)
print(x)
print(mlp:forward(x))

print(mlp.weight)
weights = torch.reshape(mlp.weight, inp)
bias = mlp.bias[1]
for i = 1, x:size(1) do
  element = x[i]
  print(element:dot(weights) + bias)
end
print("\n\n")

-- AddConstant --
t = torch.randn(3)
print(t)
m = nn.AddConstant(10, true)
print(m:forward(t))

m = nn.MulConstant(10, true)
print(m:forward(t))

-- LogSoftMax --
ii = torch.randn(10)
print(ii)
m = nn.LogSoftMax()
oo = m:forward(ii)
print(oo)