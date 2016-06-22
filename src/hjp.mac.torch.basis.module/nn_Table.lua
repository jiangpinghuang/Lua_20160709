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