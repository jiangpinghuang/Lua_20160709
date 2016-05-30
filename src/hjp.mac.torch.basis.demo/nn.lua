--[[
Containers.
--]]

require 'nn'

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