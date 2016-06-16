-- Math Functions
require('torch')

-- torch.log(x,x)
-- x:log
x = torch.rand(100, 100)
k = torch.rand(10, 10)
res1 = torch.conv2(x,k)
res2 = torch.Tensor()
print(torch.conv2(res2, x, k))
print(res2:dist(res1))

for i = 1, 100 do
  torch.conv2(res2, x, k)
end
print(res2:dist(res1))

x = torch.cat(torch.ones(3), torch.zeros(2))
print(x)
x = torch.cat(torch.ones(3, 2), torch.zeros(3, 2), 1)
print(x)
x = torch.cat(torch.ones(3, 2), torch.zeros(3, 2), 2)
print(x)
x = torch.cat(torch.cat(torch.ones(2,2), torch.zeros(2,2), 1), torch.rand(3,2),1)
print(x)
x = torch.cat({torch.ones(2,2), torch.zeros(2,2), torch.rand(3,2)}, 1)
print(x)

p = torch.Tensor{1,1,0.5,0}
a = torch.multinomial(p, 10000, true)
print(a)
for i = 1, 4 do print(a:eq(i):sum()) end

