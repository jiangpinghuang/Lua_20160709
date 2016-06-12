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