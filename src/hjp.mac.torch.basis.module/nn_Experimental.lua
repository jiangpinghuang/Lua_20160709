-- nnx: experimental 'nn' components

require 'nn'
require 'nnx'

-- SoftMaxTree
input = torch.randn(5, 10)
target = torch.IntTensor{20, 24, 27, 10, 12}
gradOutput = torch.randn(5)
root_id = 29
input_size = 10
hierarchy = {
  [29] = torch.IntTensor{30, 1, 2}, [1] = torch.IntTensor{3, 4, 5},
  [2] = torch.IntTensor{6, 7, 8}, [3] = torch.IntTensor{9, 10, 11},
  [4] = torch.IntTensor{12, 13, 14}, [5] = torch.IntTensor{15, 16, 17},
  [6] = torch.IntTensor{18, 19, 20}, [7] = torch.IntTensor{21, 22, 23},
  [8] = torch.IntTensor{24, 25, 26, 27, 28}
}
smt = nn.SoftMaxTree(input_size, hierarchy, root_id)
smt:forward{input, target}
smt:backward({input, target}, gradOutput)

-- TreeNLLCriterion
mlp = nn.Sequential()
linear = nn.Linear(50, 100)
push = nn.PushTable(2)
pull = push.pull(2)
mlp:add(push)
mlp:add(nn.SelectTable(1))
mlp:add(linear)
mlp:add(pull)
mlp:add(smt)
mlp:forward{input, target}
mlp:backward({input, target}, gradOutput)

mlp2 = nn.Sequential()
para = nn.ParallelTable()
para:add(linear)
para:add(nn.Identity())
mlp2:add(para)
mlp2:add(smt)
mlp2:forward{input, target}
mlp2:backward({input, target}, gradOutput)
 
-- SpatialReSampling
require 'image'
require 'nnx' 
input = image.loadPNG('doc/image/Lenna.png')
l = nn.SpatialReSampling{owidth=150, oheight=150}
output = l:forward(input)
image.save('doc/image/Lenna-150m150-bilinear.png', output)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  