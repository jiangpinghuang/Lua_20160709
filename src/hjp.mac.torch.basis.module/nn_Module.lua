-- Module

require 'nn'

-- accUpdateGradParameters(input, gradOutput, learningRate)
--function Module:accUpdateGradParameters(input, gradOutput, lr)
--  local gradWeight = self.gradWeight
--  local gradBias = self.gradBias
--  self.gradWeight = self.weight
--  self.gradBias = self.bias
--  self:accGradParameters(input, gradOutpu, -lr)
--  self.gradWeight = gradWeight
--  self.gradBias = gradBias
--end

-- share(mlp, s1, s2, ..., sn)
mlp1 = nn.Sequential()
mlp1:add(nn.Linear(100, 10))

mlp2 = nn.Sequential()
mlp2:add(nn.Linear(100, 10))

mlp2:share(mlp1, 'bias')
mlp1:get(1).bias[1] = 99

print(mlp2:get(1).bias[1])

-- clone(mlp, ...)
mlp1 = nn.Sequential()
mlp1:add(nn.Linear(100, 10))

mlp2 = mlp1:clone('weight', 'bias')
mlp1:get(1).bias[1] = 99

print(mlp2:get(1).bias[1])

-- type(type[, tensorCache])
mlp1 = nn.Sequential()
mlp1:add(nn.Linear(100, 10))

mlp2 = nn.Sequential()
mlp2:add(nn.Linear(100,10))
mlp2:share(mlp1, 'bias')

nn.utils.recursiveType({mlp1, mlp2}, 'torch.FloatTensor')

-- findModules(typename)
model = nn.ParallelTable()
conv1 = nn.Sequential()
conv1:add(nn.SpatialConvolution(3,16,5,5))
conv1:add(nn.Threshold())
model:add(conv1)
conv2 = nn.Sequential()
conv2:add(nn.SpatialConvolution(3,16,5,5))
conv2:add(nn.Threshold())
model:add(conv2)

input = {torch.rand(3,128,128), torch.rand(3,64,64)}
model:forward(input)
conv = model:findModules('nn.SpatialConvolution')

for i = 1, #conv do
  print(conv[i].output:size())
end

threshold_nodes, container_nodes = model:findModules('nn.Threshold')
for i = 1, #threshold_nodes do 
  for j = 1, #(container_nodes[i].modules) do
    if container_nodes[i].modules[j] == threshold_nodes[i] then
      container_nodes[i].modules[j] = nn.Tanh()
    end
  end
end

--listModules()
mlp = nn.Sequential()
mlp:add(nn.Linear(10,20))
mlp:add(nn.Tanh())

mlp2 = nn.Parallel()
mlp2:add(mlp)
mlp2:add(nn.ReLU())

for i, module in ipairs(mlp2:listModules()) do
  print(module)
end

--apply(function)
model:apply(function(module)
  module.train = true
end)

--replace(function)
model:replace(function(module)
  if torch.typename(module) == 'nn.Dropout' then
    return nn.Identity()
  else
    return module
  end
end)