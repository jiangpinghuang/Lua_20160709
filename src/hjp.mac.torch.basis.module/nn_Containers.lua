--Containers

require 'nn'

--Container,add(module), get(index), size()
--Sequential()
mlp = nn.Sequential()
mlp:add(nn.Linear(3, 5))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(5, 1))
print(mlp)
print(mlp:forward(torch.randn(3)))

--remove([index])
model = nn.Sequential()
model:add(nn.Linear(2, 3))
model:add(nn.Linear(3, 3))
model:add(nn.Linear(3, 2))
print(model)
print(model:forward(torch.randn(2)))
model:remove(2)
print(model)

--insert(module, [index])
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 30))
model:insert(nn.Linear(20, 20), 2)
print(model)
print(model:forward(torch.randn(10)))

--Parallel
mlp = nn.Parallel(2, 1)
mlp:add(nn.Linear(10, 3))
mlp:add(nn.Linear(10, 2))
input = torch.randn(10, 2)
print(input)
print(mlp:forward(input))
print(mlp)

mlp = nn.Sequential()
c = nn.Parallel(1, 2)
for i = 1, 10 do
  local t = nn.Sequential()
  t:add(nn.Linear(3, 2))
  t:add(nn.Reshape(2, 1))
  c:add(t)
end
mlp:add(c)

print(mlp)
pred = mlp:forward(torch.randn(10, 3))
print(pred)

--Train for a few iterations
for i=1, 10000 do
  x = torch.randn(10, 3)
  y = torch.ones(2, 10)
  pred = mlp:forward(x)
  
  criterion = nn.MSECriterion()
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(0.01)
  print(err)
end

--Concat
mlp = nn.Concat(1)
mlp:add(nn.Linear(5, 3))
mlp:add(nn.Linear(5, 7))
print(mlp)
print(mlp:forward(torch.randn(5)))

--DepthConcat
inputSize = 3
outputSize = 2
input = torch.randn(inputSize, 7, 7)
mlp = nn.DepthConcat(1)
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
print(mlp:forward(input))
print(mlp)