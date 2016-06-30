require 'torch'
require 'nn'

input = torch.randn(4, 3)
print(input)

local is, os, kw, dw = 3, 3, 1, nil

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
cnn:add(nn.Max(2))
cnn:add(nn.Reshape(4, 1))
print('output:')
output = cnn:forward(input)
print(output)

print('cnn test!')
input = torch.randn(3,10)
print(input)
cnn = nn.Sequential()
cnn:add(nn.TemporalConvolution(10, 10, 1, dw))
cnn:add(nn.Tanh())
print(cnn:forward(input))
cnn:add(nn.TemporalConvolution(10, 10, 2, dw))
print(cnn:forward(input))
cnn:add(nn.Tanh())
print(cnn:forward(input))
cnn:add(nn.Max(1))
print(cnn:forward(input))


inputSize = 10
outputSize = 5
input = torch.randn(inputSize,7,7)
print(input)
mlp = nn.Sequential()
--mlp=nn.DepthConcat(1);
--mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
--print(mlp)
--mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))

mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
mlp:add(nn.ReLU())
--mlp:add(nn.Max(1))
print(mlp)
print(mlp:forward(input))
mlp:add(nn.Max(1))
mlp:add(nn.Reshape(16,1))
--print(mlp)
print(mlp:forward(input))