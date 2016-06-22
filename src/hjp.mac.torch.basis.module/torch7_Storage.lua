-- Storage

x = torch.IntStorage(10):fill(1)
y = torch.DoubleStorage(10):copy(x)
print(x)
print(y)

x = torch.DoubleStorage(10)
print(x)

x = torch.IntStorage({1, 2, 3, 4})
print(x)

-- torch.TypeStorage(storage, ...)
x = torch.DoubleStorage(10)
y = torch.DoubleStorage(x, 4, 5)
x:fill(0)
y:fill(1)
print(x)
print(y)

-- torch.TypeStorage(filename, ...)
--$ echo "Hello World" > hello.txt
--$ lua
--require 'torch'
--x = torch.CharStorage('hello.txt')
--print(x)
--print(x:string())
--print(x:fill(42):string())

-- copy
x = torch.DoubleStorage(10)
print(x[5])

x = torch.IntStorage(10):fill(1)
y = torch.DoubleStorage(10):copy(x)
print(x)
print(y)

x = torch.IntStorage(10):fill(0)
print(x)

x = torch.DoubleStorage(10):fill(1)
y = torch.DoubleStorage():resize(x:size()):copy(x)
print(x)
print(y)

-- string(str)
x = torch.CharStorage():string("blah blah")
print(x)

-- string()
x = torch.CharStorage():string("blah blash")
print(x:string())