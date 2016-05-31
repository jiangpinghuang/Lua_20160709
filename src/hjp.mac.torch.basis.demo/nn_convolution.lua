--[[
Convolutional layers
--]]

require 'nn'

seg = "-------------------"

-- TemporalConvolution --
print(seg .. "TemporalConvolution" .. seg)
inp = 5; outp = 1; kw = 1; dw = 1;
mlp = nn.TemporalConvolution(inp, outp, kw, dw)
x = torch.rand(7, inp)
print("x: ")
print(x)
print("mlp:forward(x): ")
print(mlp:forward(x))

-- show the process of convolution in detail
print("mlp.weight: ")
print(mlp.weight)
print("mlp.bias: ")
print(mlp.bias)
weights = torch.reshape(mlp.weight, inp)
print("weights: ")
print(weights)
bias = mlp.bias[1]
for i = 1, x:size(1) do
  element = x[i]
  print(element:dot(weights) + bias)
end

-- LookupTable --
print(seg .. "LookupTable" .. seg)
-- 1D example
module = nn.LookupTable(10, 3)

input = torch.Tensor{1, 2, 1, 10}
print(module:forward(input))

-- 2D example
module = nn.LookupTable(10, 3)
input = torch.Tensor({{1, 2, 4, 5}, {4, 3, 2, 10}})
print(module:forward(input))

-- max-norm regularization example
module = nn.LookupTable(10, 3, 0, 1, 2)
input = torch.Tensor{1, 2, 1, 10}
print(module.weight)
print(module:forward(input))
print(module.weight)

print(seg .. "End" .. seg)