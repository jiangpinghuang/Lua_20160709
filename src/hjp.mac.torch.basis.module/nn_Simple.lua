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

-- Bilinear
mlp = nn.Sequential()
mlp:add(nn.Bilinear(10, 5, 3))
input = {torch.randn(128, 10), torch.randn(128, 5)}
print(mlp:forward(input))

-- PartialLinear
module = nn.PartialLinear(5, 3)
input = torch.randn(8, 5)
print(module:forward(input))

-- Dropout
module = nn.Dropout()
x = torch.Tensor{{1,2,3,4}, {5,6,7,8}}
print(module:forward(x))
print(module:backward(x, x:clone():fill(1)))
print(module:evaluate())
print(module:forward(x))
print(module:training())
print(module:forward(x))

-- Abs
m = nn.Abs()
ii = torch.linspace(-5, 5)
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
--gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
--gnuplot.grid(true)

-- Add
y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.Add(5))

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
  return err
end

for i = 1, 10000 do
  x = torch.rand(5)
  y:copy(x)
  for i = 1, 5 do y[i] = y[i] + i; end
  err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).bias)
mlp = nn.Sequential()
mlp:add(nn.Add(5, false))
t = torch.randn(5)
print(t)
print(mlp:forward(t))

-- Mul
y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.Mul())

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
  return err
end

for i = 1, 10000 do
  x = torch.rand(5)
  y:copy(x)
  y:mul(math.pi)
  err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).weight)

-- CMul
y = torch.Tensor(5)
mlp = nn.Sequential()
mlp:add(nn.CMul(5))
sc = torch.Tensor(5)
for i = 1, 5 do sc[i] = i; end

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
  return err
end

for i = 1, 10000 do
  x = torch.rand(5)
  y:copy(x)
  y:cmul(sc)
  err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).weight)

-- Identity
mlp = nn.Identity()
print(mlp:forward(torch.ones(5, 3)))

pred_mlp = nn.Sequential()
pred_mlp:add(nn.Linear(5, 4))
pred_mlp:add(nn.Linear(4, 3))

xy_mlp = nn.ParallelTable()
xy_mlp:add(pred_mlp)
xy_mlp:add(nn.Identity())

mlp = nn.Sequential()
mlp:add(xy_mlp)
cr = nn.MSECriterion()
cr_wrap = nn.CriterionTable(cr)
mlp:add(cr_wrap)

for i = 1, 100 do
  x = torch.ones(5)
  y = torch.Tensor(3)
  y:copy(x:narrow(1, 1, 3))
  err = mlp:forward{x, y}
  print(err)
  
  mlp:zeroGradParameters()
  mlp:backward({x, y})
  mlp:updateParameters(0.05)
end

-- Narrow
x = torch.rand(4, 5)
print(x)
print(nn.Narrow(1, 2, 3):forward(x))
print(nn.Narrow(1, 2, 1):forward(x))
print(nn.Narrow(1, 2, 2):forward(x))
print(nn.Narrow(1, 2, 1):forward(x))
print(nn.Narrow(2, 2, 3):forward(x))
print(nn.Narrow(2, 2, 2):forward(x))

-- Replicate
x = torch.linspace(1, 5, 5)
print(x)
m = nn.Replicate(3)
o = m:forward(x)
print(o)
print(x:fill(13))
print(o)

-- Reshape
x = torch.Tensor(4, 4)
for i = 1, 4 do
  for j = 1, 4 do
    x[i][j] = (i - 1) * 4 + j
  end
end
print(x)
print(nn.Reshape(2, 8):forward(x))
print(nn.Reshape(8, 2):forward(x))
print(nn.Reshape(16):forward(x))

y = torch.Tensor(1, 4):fill(0)
print(y)
print(nn.Reshape(4):forward(y))
print(nn.Reshape(4, false):forward(y))

-- View
x = torch.Tensor(4, 4)
for i = 1, 4 do
  for j = 1, 4 do
    x[i][j] = (i - 1) * 4 + j
  end
end
print(x)
print(nn.View(2, 8):forward(x))
print(nn.View(torch.LongStorage{8,2}):forward(x))
print(nn.View(16):forward(x))

input = torch.Tensor(2, 2)
minibatch = torch.Tensor(5, 2, 3)
m = nn.View(-1):setNumInputDims(2)
print(#m:forward(input))
print(#m:forward(minibatch))

-- Select
mlp = nn.Sequential()
mlp:add(nn.Select(2, 3))
x = torch.randn(10, 5)
print(x)
print(mlp:forward(x))

mlp = nn.Sequential()
c = nn.Concat(2)
for i = 1, 10 do
  local t = nn.Sequential()
  t:add(nn.Select(1, i))
  t:add(nn.Linear(3, 2))
  t:add(nn.Reshape(2, 1))
  c:add(t)
end
mlp:add(c)

pred = mlp:forward(torch.randn(10, 3))
print(pred)

for i = 1, 10000 do
  x = torch.randn(10, 3)
  y = torch.ones(2, 10)
  pred = mlp:forward(x)
  
  criterion = nn.MSECriterion()
  err = criterion:forward(pred, y)
  gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(0.01)
  print(err)
end

-- MaskedSelect
ms = nn.MaskedSelect()
mask = torch.ByteTensor({{1, 0}, {0, 1}})
input = torch.DoubleTensor({{10, 20}, {30, 40}})
print(input)
print(mask)
out = ms:forward({input, mask})
print(out)
gradIn = ms:backward({input, mask}, out)
print(gradIn[1])

-- Unsqueeze
input = torch.Tensor(2, 4, 3)
m = nn.Unsqueeze(1)
m:forward(input)

m = nn.Unsqueeze(4)
m:forward(input)

m = nn.Unsqueeze(2)
m:forward(input)

input2 = torch.Tensor(3, 5, 7)
m:forward(input2)

b = 5
input = torch.Tensor(b, 2, 4, 3)
numInputDims = 3

m = nn.Unsqueeze(4, numInputDims)
m:forward(input)

m = nn.Unsqueeze(2):setNumInputDims(numInputDims)
m:forward(input)

-- Exp
ii = torch.linspace(-5, 5)
m = nn.Exp()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- Square & Sqrt & 
ii = torch.linspace(0, 2)
m = nn.Power(1.25) -- Square() & Sqrt()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- Clamp
A = torch.randn(2, 5)
m = nn.Clamp(-0.1, 0.5)
B = m:forward(A)

print(A)
print(B)

-- Normalize
A = torch.randn(3, 5)
m = nn.Normalize(2)
B = m:forward(A)
print(A)
print(B)
print(torch.norm(B, 2, 2))

A = torch.randn(3, 5)
m = nn.Normalize(math.huge)
B = m:forward(A)
maxA = torch.abs(A):max(2)
print(A, B, maxA)

-- MM
model = nn.MM()
A = torch.randn(2, 3, 2)
B = torch.randn(2, 2, 3)
C = model:forward({A, B})
print(A)
print(B)
print(C)

model = nn.MM(true, false)
A = torch.randn(2, 3, 2)
B = torch.randn(2, 3, 3)
C = model:forward({A, B})
print(A)
print(B)
print(C)

-- BatchNormalization
model = nn.BatchNormalization(3)
A = torch.randn(5, 3)
C = model:forward(A)
print(A)
print(C)

model = nn.BatchNormalization(3, nil, nil, false)
A = torch.randn(5, 3)
C = model:forward(A)
print(A)
print(C)

-- Padding
module = nn.Padding(1, 2, 1, -1)
x = torch.randn(3)
y = module:forward(x)
print(y)

module = nn.Padding(1, -2, 1, -1)
y = module:forward(torch.randn(2,3))
print(y)

module = nn.Padding(1, -2, 1, -1, 2)
y = module:forward(torch.randn(2, 3))
print(y)

-- L1Penalty
encoder = nn.Sequential()
encoder:add(nn.Linear(3, 128))
encoder:add(nn.Threshold())
decoder = nn.Linear(128, 3)

autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(nn.L1Penalty(l1weight))
autoencoder:add(decoder)

criterion = nn.MSECriterion()