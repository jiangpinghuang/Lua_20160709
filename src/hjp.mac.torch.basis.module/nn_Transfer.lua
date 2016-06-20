-- Transfer Function Layers

require 'nn'
require 'gnuplot'

-- HardTanh, f(x) = 1, if x > 1; f(x) = -1, if x < -1; f(x) = x, otherwise
ii = torch.linspace(-2, 2)
m = nn.HardTanh(-0.5, 0.5)
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
--gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
--gnuplot.grid(true)

-- HardShrink, f(x) = x, if x > lambda; f(x) = x, if x < -lambda; f(x) = 0, otherwise
ii = torch.linspace(-2, 2)
local lambda = 0.85
m = nn.HardShrink(lambda)
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
--gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
--gnuplot.grid(true)

-- SoftShrink, f(x) = x - lambda, if x > lambda; f(x) = x + lambda, if x < -lambda; f(x) = 0, otherwise
ii = torch.linspace(-2, 2)
local lambda = 0.85
m = nn.SoftShrink(lambda)
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
--gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
--gnuplot.grid(true)

-- SoftMax, output Tensor lie in the range(0, 1) and sum to 1
x = torch.randn(10)
print(x)
y = torch.abs(x)
print(y)
ii = torch.exp(y)
print(ii)
m = nn.SoftMax()
oo = m:forward(ii)
print(oo)
--gnuplot.plot({'Input', ii, '+-'}, {'Output', oo, '+-'})
--gnuplot.grid(true)

-- SoftMin, output Tensor lie in the range(0, 1) and sum to 1
x = torch.randn(10)
print(x)
y = torch.abs(x)
print(y)
ii = torch.exp(y)
print(ii)
m = nn.SoftMin()
oo = m:forward(ii)
print(oo)
--gnuplot.plot({'Input', ii, '+-'}, {'Output', oo, '+-'})
--gnuplot.grid(true)

-- SoftPlus
ii = torch.linspace(-3, 3)
m = nn.SoftPlus()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- SoftSign, f_i(x) = x_i / (1 + |x_i|)
ii = torch.linspace(-5, 5)
print(ii)
m = nn.SoftSign()
oo = m:forward(ii)
go = torch.ones(100)
print(go)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- LogSigmoid, f_i(x) = log(1/(1+exp(-x_i)))
ii = torch.randn(10)
m = nn.LogSigmoid()
oo = m:forward(ii)
go = torch.ones(10)
gi = m:backward(ii, go)
gnuplot.plot({'Input', ii, '+-'}, {'Output', oo, '+-'}, {'gradInput', gi, '+-'})
gnuplot.grid(true)

-- LogSoftMax
ii = torch.randn(10)
m = nn.LogSoftMax()
oo = m:forward(ii)
go = torch.ones(10)
gi = m:backward(ii, go)
gnuplot.plot({'Input', ii, '+-'}, {'Output', oo, '+-'}, {'gradInput', gi, '+-'})
gnuplot.grid(true)

-- Sigmoid, defines as f(x) = 1/(1+exp(-x))
ii = torch.linspace(-5, 5)
m = nn.Sigmoid()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- Tanh, difined as f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
ii = torch.linspace(-3, 3)
m = nn.Tanh()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- ReLU, difined as f(x) = max(0, x)
ii = torch.linspace(-3, 3)
m = nn.ReLU()
oo = m:forward(ii)
go = torch.ones(100)
gi = m:backward(ii, go)
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- ReLU6, defined as f(x) = min(max(0, x), 6)
--ii=torch.linspace(-3, 9)
--m=nn.ReLU6() 
--oo=m:forward(ii)
--go=torch.ones(100)
--gi=m:backward(ii,go)
--gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
--gnuplot.grid(true)

-- RReLU, defined as f(x) = max(0, x) + a * min(0, x), where a ~U(l, u)
ii = torch.linspace(-3, 3)
m = nn.RReLU()
oo = m:forward(ii):clone()
gi = m:backward(ii, torch.ones(100))
gnuplot.plot({'f(x)', ii, oo, '+-'}, {'df/dx', ii, gi, '+-'})
gnuplot.grid(true)

-- ELU, defined as f(x) = max(0, x) + min(0, a*(exp(x) - 1))
xs = torch.linspace(-3, 3, 200)
go = torch.ones(xs:size(1))
function f(a) return nn.ELU(a):forward(xs) end
function df(a) local m = nn.ELU(a) m:forward(xs) return m:backward(xs, go) end
gnuplot.plot({'fw ELU, alpha=0.1', xs, f(0.1), '-'},
             {'fw ELU, alpha=1.0', xs, f(1.0), '-'},
             {'bw ELU, alpha=0.1', xs, df(0.1), '-'},
             {'bw ELU, alpha=1.0', xs, df(1.0), '-'})     
gnuplot.grid(true)  