-- Simple

require 'nn'

-- Linear
module = nn.Linear(10, 5)
mlp = nn.Sequential()
mlp:add(module)

print(module.weight)
print(module.bias)
print(module.gradWeight)
print(module.gradBias)

x = torch.Tensor(10)
print(x)
y = module:forward(x)
print(y)

-- SparseLinear
module = nn.SparseLinear(10000, 2)
x = torch.Tensor({{1, 0.1}, {2, 0.3}, {10, 0.3}, {31, 0.2}})
print(x)

print(module.weight)
print(module.bias)
print(module.gradWeight)
print(module.gradBias)
print(module:forward(x))