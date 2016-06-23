-- Overview

require 'nn'

-- Plug and play
mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(25, 1))

input = torch.rand(10)
print(input)
output = mlp:forward(input)
print(output)

--criterion = nn.MSECriterion()
--trainer = nn.StochasticGradient(mlp, criterion)
--trainer:train(dataset)

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
linear = nn.Linear(2, 2)
linear_clone = linear:clone('weight', 'bias')
mlp:add(linear)
mlp:add(linear_clone)
function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
linear = nn.Linear(2, 2)
linear_clone = linear:clone('weight', 'bias', 'gradWeight', 'gradBias')
mlp:add(linear)
mlp:add(linear_clone)
params, gradParams = mlp:getParameters()
function gradUpdate(mlp, x, y, criterion, learningRate, params, gradParams)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  params:add(-learningRate, gradParams)
end
