--[[
Transfer Function Layers
--]]

require 'nn'

-- HardTanh --
ii = torch.linspace(-2, 2)
print("ii: ")
print(ii)
m = nn.HardTanh()
oo = m:forward(ii)
print("oo: ")
print(oo)

-- ReLU --
--m = nn.ReLU(true)
ii = torch.linspace(-3, 3)
print("ii: ")
print(ii)
m = nn.ReLU()
oo = m:forward(ii)
print("oo: ")
print(oo)
