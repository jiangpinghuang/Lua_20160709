-- Convolutional layers

require 'nn'

-- TemporalConvolution
print('TemporalConvolution: ')
inp = 5
outp = 1
kw = 1
dw = 1

mlp = nn.TemporalConvolution(inp, outp, kw, dw)
x = torch.rand(7, inp)
print('x:')
print(x)
print(mlp:forward(x))

weights = torch.reshape(mlp.weight, 5)
print('weights: ')
print(weights)
bias = mlp.bias[1]
print('bias: ')
print(bias)
print('x:')
print(x)
print('x:size(1):')
print(x:size(1))
for i = 1, x:size(1) do
  element = x[i]
  print('element: ')
  print(element)
  print(element:dot(weights) + bias)
end

-- LookupTable
print('LookupTable: ')
module = nn.LookupTable(10, 3)

input = torch.Tensor{1, 2, 1, 10}
print(input)
print(module:forward(input))

module = nn.LookupTable(10, 3)
print(module)
input = torch.Tensor({{1, 2, 4, 5}, {4, 3, 2, 10}})
print(input)
print(module:forward(input))

module = nn.LookupTable(10, 3, 0, 1, 2)
input = torch.Tensor{1, 2, 1, 10}
print(module.weight)
print(module:forward(input))
print(module.weight)

-- SpatialSubtractiveNormalization
require 'image'
require 'nn'

--lena = image.rgb2y(image.lena())
--ker = torch.ones(11)
--m = nn.SpatialSubstractiveNormalization(1, ker)
--processed = m:forward(lena)
--w1 = image.display(lena)
--w2 = image.display(processed)

-- SpatialBatchNormalization
--model = nn.SpatialBatchNormalization(m)
--A = torch.randn(b, m, h, w)
--C = model:forward(A)
--
--model = nn.SpatialBatchNormalization(m, nil, nil, false)
--A = torch.randn(b, m, h, w)
--C = model:forward(A)




























