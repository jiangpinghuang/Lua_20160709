require 'torch'
require 'nn'

input = torch.randn(4, 5)
print(input)

local is, os, kw, dw = 5, 1, 1, 1

cnn = nn.TemporalConvolution(is, os, kw, dw)

weights = torch.reshape(cnn.weight, is) 
print(weights)

output = cnn:forward(input)
print(output)