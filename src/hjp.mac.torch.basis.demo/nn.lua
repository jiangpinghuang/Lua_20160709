require 'nn'
--[[
--||Containers.||--

-- Sequential. --
mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(25, 1))

print(mlp:forward(torch.randn(10)))

model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 20))
model:add(nn.Linear(20, 30))
print(model)

model:remove(2)
print(model)

model:insert(nn.Linear(20, 20), 2)
print(model)

-- Parallel. --
mlp = nn.Parallel(2, 1)
mlp:add(nn.Linear(10, 3))
mlp:add(nn.Linear(10, 2))
print(mlp:forward(torch.randn(10, 2)))

mlp=nn.Sequential();
c=nn.Parallel(1,2)
for i=1,10 do
  local t=nn.Sequential()
  t:add(nn.Linear(3,2))
  t:add(nn.Reshape(2,1))
  c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- Train for a few iterations
  x=torch.randn(10,3);
  y=torch.ones(2,10);
  pred=mlp:forward(x)

  criterion= nn.MSECriterion()
  local err=criterion:forward(pred,y)
  local gradCriterion = criterion:backward(pred,y);
  mlp:zeroGradParameters();
  mlp:backward(x, gradCriterion); 
  mlp:updateParameters(0.01);
  print(err)
end

-- Concat. --
module = nn.Concat(dim)

mlp=nn.Concat(1)
mlp:add(nn.Linear(5, 3))
mlp:add(nn.Linear(5, 7))
print(mlp:forward(torch.randn(5)))

-- DepthConcat. --
module = nn.DepthConcat(dim)

inputSize = 3
outputSize = 2
input = torch.randn(inputSize,7,7)
mlp=nn.DepthConcat(1);
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
print(mlp:forward(input))
--]]


--//////////////////////////////////////////--
--[[
--||Overview.||--
--]]
-- A simple neural network (perceptron) with 10 inputs. --
-- mlp = nn.Linear(10, 1)

-- Multi-layer perceptron. --
mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(25, 1))

print(mlp)

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
-- No dataset. --
--trainer:train(dataset)

function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

-- our optimization procedure will iterate over the modules, so only share
-- the parameters
mlp = nn.Sequential()
linear = nn.Linear(2,2)
linear_clone = linear:clone('weight','bias') -- clone sharing the parameters
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

-- our optimization procedure will use all the parameters at once, because
-- it requires the flattened parameters and gradParameters Tensors. Thus,
-- we need to share both the parameters and the gradParameters
mlp = nn.Sequential()
linear = nn.Linear(2,2)
-- need to share the parameters and the gradParameters as well
linear_clone = linear:clone('weight','bias','gradWeight','gradBias')
mlp:add(linear)
mlp:add(linear_clone)
params, gradParams = mlp:getParameters()
function gradUpdate(mlp, x, y, criterion, learningRate, params, gradParams)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  -- adds the gradients to all the parameters at once
  params:add(-learningRate, gradParams)
end


--//////////////////////////////////////////--
--[[
--||Transfer Function Layers.||--
--]]

require 'gnuplot'
-- HardTanh. --
ii = torch.linspace(-2, 2)
m = nn.HardTanh()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

module = nn.HardShrink(lambda)
ii=torch.linspace(-2,2)
m=nn.HardShrink(0.85)
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)