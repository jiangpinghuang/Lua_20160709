require 'torch'
require 'nn'

input = torch.randn(4, 3)
print(input)

local is, os, kw, dw = 3, 3, 1, 1

cnn = nn.Sequential()
cnn:add(nn.TemporalConvolution(is, os, kw, dw))
cnn:add(nn.ReLU())
--cnn = nn.TemporalConvolution(is, os, kw, dw)

print(cnn.weight)
print(cnn.bias)
--weights = torch.reshape(cnn.weight, is) 
--print(weights)
print('output:')
output = cnn:forward(input)
print(output)