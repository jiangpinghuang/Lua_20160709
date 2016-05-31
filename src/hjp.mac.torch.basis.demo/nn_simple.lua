--[[
Simple layers
--]]

require 'nn'

-- Linear --
-- 10 inputs, 5 outputs
module = nn.Linear(10, 5) 

mlp = nn.Sequential()
mlp:add(module)
print("module.weight: ")
print(module.weight)
print("module.bias: ")
print(module.bias)
print("module.gradWeight: ")
print(module.gradWeight)
print("module.gradBias: ")
print(module.gradBias)

x = torch.Tensor(10)
print("module:forward(x): ")
print(module:forward(x))

