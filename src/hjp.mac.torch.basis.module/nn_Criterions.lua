-- Criterions

require 'nn'

-- ClassNLLCriterion
function gradUpdate(mlp, x, y, learningRate)
  local criterion = nn.ClassNLLCriterion()
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  mlp:zeroGradParameters()
  local t = criterion:backward(pred, y)
  mlp:backward(x, t)
  mlp:updateParameters(learningRate)
end

-- ClassSimplexCriterion
--nInput = 10
--nClasses = 30
--nHidden = 100
--mlp = nn.Sequential()
--mlp:add(nn.Linear(nInput, nHidden)):add(nn.ReLU())
--mlp:add(nn.NormalizedLinearNoBias(nHidden, nClasses))
--mlp:add(nn.Normalize(2))
--
--criterion = nn.ClassSimplexCriterion(nClasses)
--
--function gradUpdate(mlp, x, y, learningRate)
--  local pred = mlp:forward(x)
--  local err = criterion:forward(pred, y)
--  mlp:zeroGradParameters()
--  local t = criterion:backward(pred, y)
--  mlp:backward(x, t)
--  mlp:updateParameters(learningRate)
--end

-- MarginCriterion
print('MarginCriterion: ')

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(5, 1))

x1 = torch.rand(5)
x1_target = torch.Tensor{1}
x2 = torch.rand(5)
x2_target = torch.Tensor{-1}
criterion = nn.MarginCriterion(1)

for i = 1, 1000 do
  gradUpdate(mlp, x1, x1_target, criterion, 0.01)
  gradUpdate(mlp, x2, x2_target, criterion, 0.01)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1), x1_target))  
print(criterion:forward(mlp:forward(x2), x2_target))
  
 -- SoftMarginCriterion
function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(5, 1))

x1 = torch.rand(5)
x1_target = torch.Tensor{1}
x2 = torch.rand(5)
x2_target = torch.Tensor{-1}
criterion = nn.SoftMarginCriterion(1) 

for i = 1, 1000 do
  gradUpdate(mlp, x1, x1_target, criterion, 0.01)
  gradUpdate(mlp, x2, x2_target, criterion, 0.01)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1), x1_target))  
print(criterion:forward(mlp:forward(x2), x2_target)) 
  
-- MultiLabelMarginCriterion
print("MultiLabelMarginCriterion: ")
criterion = nn.MultiLabelMarginCriterion()
input = torch.randn(2, 4)
target = torch.Tensor{{1, 3, 0, 0}, {4, 0, 0, 0}}
print(criterion:forward(input, target))
  
-- MultiCriterion
print('MultiCriterion: ')
input = torch.rand(2, 10)
target = torch.IntTensor{1, 8}
nll = nn.ClassNLLCriterion()
nll2 = nn.CrossEntropyCriterion()
mc = nn.MultiCriterion():add(nll, 0.5):add(nll2)
output = mc:forward(input, target)
print(output)

-- ParallelCriterion
print('ParallelCriterion: ')
input = {torch.rand(2, 10), torch.randn(2, 10)}
target = {torch.IntTensor{1, 8}, torch.randn(2, 10)}
nll = nn.ClassNLLCriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
output = pc:forward(input, target)
print(output)

-- HingeEmbeddingCriterion
print("HingeEmbeddingCriterion:")
p1_mlp = nn.Sequential()
p1_mlp:add(nn.Linear(5, 2))
p2_mlp = nn.Sequential()
p2_mlp:add(nn.Linear(5, 2))
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

pr1 = nn.ParallelTable()
pr1:add(p1_mlp)
pr1:add(p2_mlp)

mlp = nn.Sequential()
mlp:add(pr1)
mlp:add(nn.PairwiseDistance(1))

crit = nn.HingeEmbeddingCriterion(1)

x = torch.rand(5)
y = torch.rand(5)

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

for i = 1, 10 do
  gradUpdate(mlp, {x, y}, 1, crit, 0.01)
  print(mlp:forward({x, y})[1])
end

for i = 1, 10 do
  gradUpdate(mlp, {x, y}, -1, crit, 0.01)
  print(mlp:forward({x, y})[1])
end

-- MarginRankingCriterion
print("MarginRankingCriterion:")
p1_mlp = nn.Linear(5, 2)
p2_mlp = p1_mlp:clone('weight', 'bias')

pr1 = nn.ParallelTable()
pr1:add(p1_mlp)
pr1:add(p2_mlp)

mlp1 = nn.Sequential()
mlp1:add(pr1)
mlp1:add(nn.DotProduct())

mlp2 = mlp1:clone('weight', 'bias')

mlpa = nn.Sequential()
pr1a = nn.ParallelTable()
pr1a:add(mlp1)
pr1a:add(mlp2)
mlpa:add(pr1a)

crit = nn.MarginRankingCriterion(0.1)

x = torch.randn(5)
y = torch.randn(5)
z = torch.randn(5)

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

for i = 1, 100 do
  gradUpdate(mlpa, {{x, y}, {x, z}}, 1, crit, 0.01)
  if true then
    o1 = mlp1:forward{x, y}[1]
    o2 = mlp2:forward{x, z}[1]
    o = crit:forward(mlpa:forward{{x, y}, {x, y}}, 1)
    print(o1, o2, o)
  end
end

print "--"

for i = 1, 100 do
  gradUpdate(mlpa, {{x, y}, {x, z}}, -1, crit, 0.01)
  if true then 
    o1 = mlp1:forward{x, y}[1]
    o2 = mlp2:forward{x, z}[1]
    o = crit:forward(mlpa:forward{{x, y}, {x, z}}, -1)
    print(o1, o2, o)
  end
end
