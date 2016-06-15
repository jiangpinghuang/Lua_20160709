-- create a 4D-tensor 4*5*6*2
z = torch.Tensor(4,5,6,2)
z:fill(1)
print(z)

-- more dimensions, for example 6D tensor
s = torch.LongStorage(6)
s[1] = 4; s[2] = 5; s[3] = 6; s[4] = 2; s[5] = 7; s[6] = 3;
x = torch.Tensor(s)
print(x)

-- return the number of dimension of a Tensor
print(x:nDimension())
print(x:size())

-- internal data representation
x = torch.Tensor(7,7,7)
x:fill(9)
print(x)
print(x[3][4][5])
print(x[3][4])

-- create 2D tensor
y = torch.Tensor(2,3)
y[1]:fill(2)
print(y)
print(y[1])

-- Tensor is a particular way of viewing a Storage
x = torch.Tensor(4,5)
s = x:storage()
for i=1,s:size() do
  s[i] = i
end
print(x)

-- elements are contiguous in memory for a matrix
x = torch.Tensor(4,5)
i = 0

x:apply(function()
  i = i + 1
  return i
end)

print(x)
print(x:stride())

-- Tensors of different types. Don't save memory space to use FloatTensor and DoubleTensor
x = torch.ByteTensor(3,4)
x:fill(100)
print(x)

x = torch.CharTensor(2,3)
x:fill(10)
print(x)

x = torch.DoubleTensor(2,3)
x:fill(100.00)
print(x)

-- Default Tensor type
torch.setdefaulttensortype('torch.FloatTensor')

-- Efficient memory management
x = torch.Tensor(5):zero()
print(x)

x:narrow(1, 2, 3):fill(1)
print(x)

y = torch.Tensor(x:size()):copy(x)
print(y)

y = x:clone()
print('clone!')
print(y)

-- torch.Tensor(tensor)
x = torch.Tensor(2,5):fill(3.14)
print(x)

y = torch.Tensor(x)
print(y)

y:zero()
print(x)

-- The LongStorage sizes gives the size in each dimension of the tensor
x = torch.Tensor(torch.LongStorage({4,4,3,2}))
x = torch.Tensor(torch.LongStorage({4}), torch.LongStorage({0})):zero()
x[1] = 1
print(x)

a = torch.LongStorage({1,2})
b = torch.FloatTensor(a)
print(b:size())
c = torch.LongTensor(a)
print(c)

s = torch.Storage(10):fill(1)
x = torch.Tensor(s, 1, torch.LongStorage{2,5})
print(x)
print(x:zero())
print(s)

-- torch.Tensor(table)
x = torch.Tensor({{1,2,3,4}, {5,6,7,8}})
print(x)

-- Cloning, returns a clone of a tensor, the memory is copied
i = 0
x = torch.Tensor(10):apply(function(x)
  i = i + 1
  return i
end)
print(x)

y = x:clone()
print(y)

y:fill(1)
print(y)
print(x)

-- contiguous
x = torch.Tensor(2,3):fill(1)
print(x)

y = x:contiguous():fill(2)
print(y)
print(x)

z = x:t():contiguous():fill(3.14)
print(z)
print(x)

-- Tensor or string type
print(torch.Tensor():type())

x = torch.Tensor(3):fill(3.14)
print(x)
y = x:type('torch.DoubleTensor')
print(y)
y:zero()
print(x)

x = torch.Tensor(3):fill(3.14)
print(x)
y = x:type('torch.IntTensor')
print(y)

-- boolean isTensor(object)
print(torch.isTensor(torch.randn(3,4)))
print(torch.isTensor(torch.randn(3,4)[1]))
print(torch.isTensor(torch.randn(3,4)[1][2]))

-- byte(), char(), short() ...
x = torch.Tensor(3):fill(3.14)
print(x)
print(x:type('torch.IntTensor'))
print(x:int())

-- query the size and structure
x = torch.Tensor(4,5)
print(x:nDimension())
x = torch.Tensor(4,5):zero()
print(x)

-- gets the number of columns and rows
print(x:size(2))
print(x:size())
print(x:size(1))

-- returns the jump necessary to go from one element to the next one
x = torch.Tensor(4,5):fill(1.0)
print(x)
-- elements in a column are contiguous in memory
print(x:stride(2))
-- we need here to jump the size of the column
print(x:stride(1))
print(x:stride())

-- a tensor is a particular way of viewing a storage
x = torch.Tensor(4,5)
s = x:storage()
for i = 1, s:size() do
  s[i] = i
end
print(x)

-- Return true iff the elements of the Tensor are contiguous in memory
x = torch.randn(4,5)
print(x:isContiguous())
print(x)

y = x:select(2,3)  
print(y:isContiguous())
print(y)
print(y:stride())

-- Return true iff the dimensions of the Tensor match the elements of the storage
x = torch.Tensor(4,5)
y = torch.LongStorage({4,5})
z = torch.LongStorage({5,4,1})
print(y)
print(z)
print(x:isSize(y))
print(x:isSize(z))
print(x:isSize(x:size()))

-- Return true iff the dimensions of Tensor and the argument Tensor are exactly the same
x = torch.Tensor(4,5)
y = torch.Tensor(4,5)
print(x:isSameSizeAs(y))
y = torch.Tensor(4,6)
print(x:isSameSizeAs(y))

-- Returns the number of elements of a tensor
x = torch.Tensor(4,5)
print(x:nElement())

-- Querying elements
x = torch.Tensor(3,3)
i = 0; x:apply(function() i = i + 1; return i end)
print(x)
print(x[2])
print(x[2][3])
print(x:select(1,2))
print(x:select(2,2))
print(x[{2,3}])
print(x[torch.LongStorage{2,3}])
print(x[torch.le(x,3)])

-- Referencing a tensor to an existing tensor or chunk of memory
y = torch.Storage(10)
x = torch.Tensor()
print(x:set(y, 1, 10))
y = torch.Storage(10)
x = torch.Tensor(y, 1, 10)
print(y)
print(x)

-- self set tensor
x = torch.Tensor(2,5):fill(3.14)
print(x)
y = torch.Tensor():set(x)
print(y)
print(y:zero())

-- Return true iff the Tensor is set to the argument Tensor
x = torch.Tensor(2,5)
y = torch.Tensor()
print(y:isSetTo(x))
print(y:set(x))
print(y:isSetTo(x))
print(y:t():isSetTo(x))

-- set storage
s = torch.Storage(10):fill(1)
sz = torch.LongStorage({2,5})
x = torch.Tensor()
x:set(s, 1, sz)
print(x)
x:zero()
print(s)

-- Copying and initializing
x = torch.Tensor(4):fill(1)
y = torch.Tensor(2,2):copy(x)
print(x)
print(y)
x = torch.DoubleTensor(4):fill(3.14)
print(x)
x = torch.Tensor(4):zero()
print(x)

-- For method narrow, select and sub the returned tensor shares the same storage as the original
x = torch.Tensor(5,6):zero()
print(x)

y = x:narrow(1,2,3)
y:fill(1)
print(y)
print(x)

x = torch.Tensor(5,6):zero()
print(x)
y = x:sub(2,4):fill(1)
print(y)
print(x)
z = x:sub(2,4,3,4):fill(2)
print(z)
print(x)
print(y)
print(y:sub(-1,-1,3,4))

x = torch.Tensor(5,6):zero()
print(x)
y = x:select(1,2):fill(2)
print(y)
print(x)
z = x:select(2,5):fill(5)
print(z)
print(x)

-- The indexing operator [] can be used to combine narrow/sub and select in a concise and efficient way.
x = torch.Tensor(5,6):zero()
print(x)
x[{1,3}]=1
print(x)
x[{2,{2,4}}] = 2
print(x)
x[{{}, 4}] = -1
print(x)
x[{{},2}] = torch.range(1,5)
print(x)
x[torch.lt(x,0)] = -2
print(x)

-- Tensor index(dim, index)
x = torch.rand(5,5)
print(x)
y = x:index(1, torch.LongTensor{3,1})
print(y)
y:fill(1)
print(x)

x = torch.rand(5,5)
print(x)
y = torch.Tensor()
y:index(x, 1, torch.LongTensor{3,1})  -- index(x, 1 or 2, torch.LongTensor{3,1}), 1 is row, 2 is column
print(y)

-- indexCopy(dim, index, tensor)
print(x)
z = torch.Tensor(2,5)
z:select(1,1):fill(-1)
z:select(1,2):fill(-2)
print(z)
x:indexCopy(1, torch.LongTensor{3,4}, z)
print(x)

-- indexAdd(dim, index, tensor)
print(x)
z = torch.Tensor(5,2)
z:select(2,1):fill(-1)
z:select(2,2):fill(-2)
print(z)
x:indexAdd(2,torch.LongTensor{5,1},z)
print(x)

a = torch.range(1,5)
print(a)
a:indexAdd(1, torch.LongTensor{1,1,3,3}, torch.range(1,4))
print(a)

-- indexFill(dim, index, val)
x = torch.rand(5,5)
print(x)
x:indexFill(2, torch.LongTensor{1,2}, -10)
print(x)

-- tensor gather(dim, index)
x = torch.rand(5,5)
print(x)
y = x:gather(1, torch.LongTensor{{1,2,3,4,5},{2,3,4,5,1}})
print(y)
z = x:gather(2, torch.LongTensor{{1,2},{2,3},{3,4},{4,5},{5,1}})
print(z)

-- tensor scatter(dim, index, src|val)
x = torch.rand(2,5)
print(x)
y = torch.zeros(3,5):scatter(1, torch.LongTensor{{1,2,3,1,1,},{3,1,1,2,3}},x)
print(y)
z = torch.zeros(2,4):scatter(2,torch.LongTensor{{1},{3}}, 1.23)
print(z)

-- tensor maskedSelect(mask)
x = torch.range(1,12):double():resize(3,4)
print(x)
mask = torch.ByteTensor(2,6):bernoulli()
print(mask)
y = x:maskedSelect(mask)
print(y)
z = torch.DoubleTensor()
z:maskedSelect(x, mask)
print(z)

-- tensor maskedCopy(mask, tensor)
x = torch.range(1,8):double():resize(2,4)
print(x)
mask = torch.ByteTensor(1,8):bernoulli()
print(mask)
y = torch.DoubleTensor(2,4):fill(-1)
print(y)
y:maskedCopy(mask,x)
print(y)

-- maskedFill
x = torch.range(1,4):double():resize(1,4)
print(x)
mask = torch.ByteTensor(2,2):bernoulli()
print(mask)
x:maskedFill(mask, -1)
print(x)

-- Search nonzero(tensor)
x = torch.rand(4,4):mul(3):floor():int()
print(x)
print(torch.nonzero(x))
print(x:nonzero())
indices = torch.LongTensor()
print(x.nonzero(indices, x))
print(x:eq(2):nonzero())

-- Expanding/Replicating/Squeezing tensors
x = torch.rand(10,1)
print(x)
y = torch.expand(x,10,2)
print(y)
y:fill(1)
print(y)
print(x)
i = 0; y:apply(function() i=i+1; return i end)    -- I can't understand it
print(y)
print(x)

-- repeatTensor
x = torch.rand(5)
print(x)
print(torch.repeatTensor(x,3,2))
print(torch.repeatTensor(x,3,2,1))

-- squeeze(dim)
x = torch.rand(2,1,2,1,2)
print(x)
print(torch.squeeze(x))
print(torch.squeeze(x,2))

-- view([result,] tensor, sizes)
x = torch.zeros(4)
print(x:view(2,2))
print(x:view(2,-1))
print(x:view(torch.LongStorage{2,2}))
print(x)
x = torch.rand(2,3)
print(x)
print(x:view(6))
print(x)

-- viewAs()
x = torch.zeros(4)
y = torch.Tensor(2,2)
print(x:viewAs(y))

-- transpose
x = torch.Tensor(3,4):zero()
x:select(2,3):fill(7)
print(x)
y = x:transpose(1,2)
print(y)
y:select(2,3):fill(8)
print(y)
x = y:transpose(1,2)
print(x)

-- t()
x = torch.Tensor(3,4):zero()
x:select(2,3):fill(7)
y = x:t()
print(y)
print(x)

-- permute()
x = torch.Tensor(3,4,2,5)
print(x)
print(x:size())
y = x:permute(2,3,1,4)    -- I'm sorry, I can't understand it
print(y:size())

-- unfold()
x = torch.Tensor(7)
for i = 1, 7 do x[i] = i end
print(x)
print(x:unfold(1,3,1))  -- 3:column, 1:stripe for row
print(x)
print(x:unfold(1,4,2))

-- Applying a function to a tensor
i = 0
z = torch.Tensor(3,3)
z:apply(function(x)
  i = i + 1
  return i
end)
print(z)
z:apply(math.sin)
print(z)
sum = 0
z:apply(function(x) sum = sum + x end)
print(sum)
print(z:sum())
print(z:mean())
x = torch.randn(4)
print(x)
print(x:mean())

-- map
x = torch.Tensor(3,3)
y = torch.Tensor(9)
i = 0
x:apply(function() i = i + 1; return i end)
i = 0
y:apply(function() i = i + 1; return i end)
print(x)
print(y)
x:map(y, function(xx, yy) return xx * yy end)
print(x)

-- map2
x = torch.Tensor(3,3)
y = torch.Tensor(9)
z = torch.Tensor(3,3)
i = -1; x:apply(function() i = i + 1; return math.cos(i)*math.cos(i) end)
i = 0; y:apply(function() i = i + 1; return i end)
i = 0; z:apply(function() i = i + 1; return i end)
print(x)
print(y)
print(z)
x:map2(y, z, function(xx, yy, zz) return xx+yy*zz end)
print(x)

-- split()
x = torch.randn(3,4,5)
print(x)
print(x:split(2,1))   -- I still don't understand it
print(x:split(5,2))
print(x:split(2,3))
print(x:split(2,3)[3])

-- chunk
x = torch.randn(3,4,5)
print(x)
print(x:chunk(2,1))
print(x:chunk(2,2))
print(x:chunk(2,3))

-- LuaJIT FFI access
t = torch.randn(3,2)
print(t)
t_data = torch.data(t)
print(t_data)
for i = 0, t:nElement()-1 do t_data[i] = 0 end
print(t)

t = torch.randn(3,2)
t_noncontiguous = t:transpose(1,2)
t_tran_and_con = t_noncontiguous:contiguous()
data = torch.data(t_tran_and_con)
t = torch.randn(10)
p = tonumber(torch.data(t, true))
s = torch.Storage(10, p)
tt = torch.Tensor(s)
print(tt)