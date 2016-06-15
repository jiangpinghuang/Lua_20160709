require 'nn'
require 'torch'
require 'gnuplot'

ii=torch.linspace(-2,2)
m=nn.HardTanh()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)