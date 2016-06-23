-- Table Layers

require 'nn'

-- table
mlp = nn.Sequential()
t = {x, y, z}
pred = mlp:forward(t)
pred = mlp:forward{x, y, z}

-- ConcatTable
mlp = nn.ConcatTable()
mlp:add(nn.Linear(5, 2))
mlp:add(nn.Linear(5, 3))

pred = mlp:forward(torch.randn(5))
for i, k in ipairs(pred) do print(i, k) end

mlp = nn.ConcatTable()
mlp:add(nn.Identity())
mlp:add(nn.Identity())

pred = mlp:forward{torch.randn(2), {torch.randn(3)}}
print(pred)

-- ParallelTable
mlp = nn.ParallelTable()
mlp:add(nn.Linear(10, 2))
mlp:add(nn.Linear(5, 3))

x = torch.randn(10)
y = torch.rand(5)

pred = mlp:forward{x, y}
for i, k in pairs(pred) do print(i, k) end

-- SplitTable
mlp = nn.SplitTable(1)
x = torch.randn(4, 3)
print(x)
pred = mlp:forward(x)
for i, k in ipairs(pred) do print(i, k) end

mlp = nn.SplitTable(2)
x = torch.randn(4, 3)
print(x)
pred = mlp:forward(x)
for i, k in ipairs(pred) do print(i, k) end

mlp = nn.SplitTable(1, 2)
x  = torch.randn(2, 4, 3)
print('hi: ')
print(x)
pred = mlp:forward(x)
for i, k in ipairs(pred) do print(i, k) end
x = torch.randn(4, 3)
print(x)
pred = mlp:forward(x)
for i, k in ipairs(pred) do print(i, k) end

m = nn.SplitTable(-2)
x = torch.randn(3, 2)
print(x)
out = m:forward(x)
for i, k in ipairs(out) do print(i, k) end
x = torch.randn(1, 3, 2)
print(x)
out = m:forward(x)
for i, k in ipairs(out) do print(i, k) end

print(torch.rand(5, 6))
print(torch.randn(5, 6))

mlp = nn.Sequential()
mlp:add(nn.SplitTable(2))
c = nn.ParallelTable()
c:add(nn.Linear(10, 3))
c:add(nn.Linear(10, 7))
mlp:add(c)
p = nn.ParallelTable()
p:add(nn.Linear(3, 2))
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))

pred = mlp:forward(torch.randn(10, 2))
print(pred)

for i = 1, 100 do
  x = torch.ones(10, 2)
  y = torch.Tensor(3)
  y:copy(x:select(2, 1):narrow(1, 1, 3))
  pred = mlp:forward(x)
  
  criterion = nn.MSECriterion()
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(0.05)
  
  print(err)
end

-- JoinTable
print("JoinTable: ")
x = torch.randn(5, 1)
y = torch.randn(5, 1)
z = torch.randn(2, 1)
print(x)
print(y)
print(z)

print(nn.JoinTable(1):forward{x, y})
print(nn.JoinTable(2):forward{x, y})
print(nn.JoinTable(1):forward{x, z})

module = nn.JoinTable(2, 2)
x = torch.randn(3, 1)
y = torch.randn(3, 1)
print(x)
print(y)

mx = torch.randn(2, 3, 1)
my = torch.randn(2, 3, 1)
print(mx)
print(my)

print(module:forward{x, y})
print(module:forward{mx, my})

mlp = nn.Sequential()
c = nn.ConcatTable()
c:add(nn.Linear(10, 3))
c:add(nn.Linear(10, 7))
mlp:add(c)
p = nn.ParallelTable()
p:add(nn.Linear(3, 2))
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))

pred = mlp:forward(torch.randn(10))
print(pred)

for i = 1, 100 do
  x = torch.ones(10)
  y = torch.Tensor(3)
  y:copy(x:narrow(1, 1, 3))
  pred = mlp:forward(x)
  
  criterion = nn.MSECriterion()
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(0.05)
  
  print(err)
end

-- MixtureTable
--experts = nn.ConcatTable()
--for i = 1, n do
--  local expert = nn.Sequential()
--  expert:add(nn.Linear(3, 4))
--  expert:add(nn.Tanh())
--  expert:add(nn.Linear(4, 5))
--  expert:add(nn.Tanh())
--  experts:add(expert)
--end
--
--gater = nn.Sequential()
--gater:add(nn.Linear(3, 7))
--gater:add(nn.Tanh())
--gater:add(nn.Linear(7, n))
--gater:add(nn.SoftMax())
--
--trunk = nn.ConcatTable()
--trunk:add(gater)
--trunk:add(experts)
--
--moe = nn.Sequential()
--moe:add(trunk)
--moe:add(nn.MixtureTable())
--
--out = moe:forward(torch.randn(2, 3))
--print(out)
--
--experts = nn.Concat(5)
--for i = 1, n do
--  local expert = nn.Sequential()
--  expert:add(nn.Linear(3, 4))
--  expert:add(nn.Tanh())
--  expert:add(nn.Linear(4, 4*2*5))
--  expert:add(nn.Tanh())
--  expert:add(nn.Reshape(4, 2, 5, 1))
--  experts:add(expert)
--end
--
--gater = nn.Sequential()
--gater:add(nn.Linear(3, 7))
--gater:add(nn.Tanh())
--gater:add(nn.Linear(7, n))
--gater:add(nn.SoftMax())
--
--trunk = nn.ConcatTable()
--trunk:add(gater)
--trunk:add(experts)
--
--moe = nn:Sequential()
--moe:add(trunk)
--moe:add(nn.MixtureTable(5))
--
--out = moe:forward(torch.randn(2, 3)):size()
--print(out)

-- SelectTable
input = {torch.randn(2, 3), torch.randn(2, 1)}
print(nn.SelectTable(-1):forward(input))

print(table.unpack(nn.SelectTable(1):backward(input, torch.randn(2, 3))))

input = {torch.randn(2, 3), {torch.randn(2, 1), {torch.randn(2, 2)}}}
print(nn.SelectTable(2):forward(input))
print(table.unpack(nn.SelectTable(2):backward(input, {torch.randn(2, 1), {torch.randn(2, 2)}})))
gradInput = nn.SelectTable(1):backward(input, torch.randn(2, 3))
print(gradInput)
print(gradInput[1])
print(gradInput[2][1])
print(gradInput[2][2][1])

-- NarrowTable
input = {torch.randn(2, 3), torch.randn(2, 1), torch.randn(1, 2)}
print(nn.NarrowTable(2, 2):forward(input))
print(nn.NarrowTable(1):forward(input))
print(table.unpack(nn.NarrowTable(1, 2):backward(input, {torch.randn(2, 3), torch.randn(2, 1)})))

-- FlattenTable
x = {torch.rand(1), {torch.rand(2), {torch.rand(3)}}, torch.rand(4)}
print(x)
print(nn.FlattenTable():forward(x))

-- PairwiseDistance
mlp_l1 = nn.PairwiseDistance(1)
mlp_l2 = nn.PairwiseDistance(2)
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp_l1:forward({x, y}))
print(mlp_l2:forward({x, y}))

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

-- DotProduct
mlp = nn.DotProduct()
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp:forward({x, y}))

mlp1 = nn.Linear(5, 10)
mlp2 = mlp1:clone('weight', 'bias')

pr1 = nn.ParallelTable()
pr1:add(mlp1)
pr1:add(mlp2)

mlp1 = nn.Sequential()
mlp1:add(pr1)
mlp1:add(nn.DotProduct())

mlp2 = mlp1:clone('weight', 'bias')

mlp = nn.Sequential()
prla = nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlp:add(prla)

x = torch.rand(5)
y = torch.rand(5)
z = torch.rand(5)

print(mlp1:forward{x, x})
print(mlp1:forward{x, y})
print(mlp1:forward{y, y})

crit = nn.MarginRankingCriterion(1)

function gradUpdate(mlp, x, y, criterion, learningRate)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end

inp = {{x, y}, {x, z}}

math.randomseed(1)

for i = 1, 100 do
  gradUpdate(mlp, inp, 1, crit, 0.05)
  o1 = mlp1:forward{x, y}[1]
  o2 = mlp2:forward{x, z}[1]
  o = crit:forward(mlp:forward{{x, y}, {x, z}}, 1)
  print(o1, o2, o)
end

print "_________________**"

for i = 1, 100 do
  gradUpdate(mlp, inp, -1, crit, 0.05)
  o1 = mlp1:forward{x, y}[1]
  o2 = mlp2:forward{x, z}[1]
  o = crit:forward(mlp:forward{{x, y}, {x, z}}, -1)
  print(o1, o2, o)
end

-- CosineDistance
mlp = nn.CosineDistance()
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp:forward({x, y}))

mlp = nn.CosineDistance()
x = torch.Tensor({{1, 2, 3}, {1, 2, -3}})
y = torch.Tensor({{4, 5, 6}, {-4, 5, 6}})
print(x)
print(y)
print(mlp:forward({x, y}))

p1_mlp = nn.Sequential()
p1_mlp:add(nn.Linear(5, 2))
p2_mlp = p1_mlp:clone('weight', 'bias')
pr1 = nn.ParallelTable()
pr1:add(p1_mlp)
pr1:add(p2_mlp)

mlp = nn.Sequential()
mlp:add(pr1)
mlp:add(nn.CosineDistance())

x = torch.rand(5)
y = torch.rand(5)

function gradUpdate(mlp, x, y, learningRate)
  local pred = mlp:forward(x)
  if pred[1] * y < 1 then
    gradCriterion = torch.Tensor({-y})
    mlp:zeroGradParameters()
    mlp:backward(x, gradCriterion)
    mlp:updateParameters(learningRate)
  end
end

for i = 1, 1000 do
  gradUpdate(mlp, {x, y}, 1, 0.1)
  if ((i%100) == 0) then print(mlp:forward({x, y})[1]); end
end

for i = 1, 1000 do
  gradUpdate(mlp, {x, y}, -1, 0.1)
  if((i%100) == 0) then print(mlp:forward({x, y})[1]); end
end
  
-- CriterionTable
mlp = nn.CriterionTable(nn.MSECriterion())
x = torch.randn(5)
y = torch.randn(5)
print(mlp:forward{x, x})
print(mlp:forward{x, y})

function table.print(t)
  for i, k in pairs(t) do print(i, k); end
end

mlp = nn.Sequential()
main_mlp = nn.Sequential()
main_mlp:add(nn.Linear(5, 4))
main_mlp:add(nn.Linear(4, 3))
cmlp = nn.ParallelTable()
cmlp:add(main_mlp)
cmlp:add(nn.Identity())
mlp:add(cmlp)
mlp:add(nn.CriterionTable(nn.MSECriterion()))

for i = 1, 20 do
  x = torch.ones(5)
  y = torch.Tensor(3)
  y:copy(x:narrow(1, 1, 3))
  err = mlp:forward{x, y}
  print(err)
  
  mlp:zeroGradParameters()
  mlp:backward({x, y})
  mlp:updateParameters(0.05)
end

-- CAddTable
print('CAddTable:')
ii = {torch.ones(5), torch.ones(5)*2, torch.ones(5)*3}
print(ii)
print(ii[1])
print(ii[2])
print(ii[3])

m = nn.CAddTable(true)
o = m:forward(ii)
print(o)
print(ii[1])

-- CSubTable
m = nn.CSubTable()
o = m:forward({torch.ones(5) * 2.2, torch.ones(5)})
print(o)

-- CMulTable
ii = {torch.ones(5) * 2, torch.ones(5) * 3, torch.ones(5) * 4}
m = nn.CMulTable()
o = m:forward(ii)
print(o)

-- CDivTalbe
m = nn.CDivTable()
ii = {torch.ones(5) * 2, torch.ones(5) * 0.4}
o = m:forward(ii)
print(o)