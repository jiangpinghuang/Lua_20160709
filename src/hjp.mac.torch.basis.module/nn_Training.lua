-- Training a neural network

require 'nn'

mlp = nn.Sequential()
inputs = 2
outputs = 1
HUs = 20
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

criterion = nn.MSECriterion()

for i = 1, 2500 do
  local input = torch.randn(2)
  local output = torch.Tensor(1)
  if input[1] * input[2] > 0 then
    output[1] = -1
  else
    output[1] = 1
  end
  
  criterion:forward(mlp:forward(input), output)
  
  mlp:zeroGradParameters()
  mlp:backward(input, criterion:backward(mlp.output, output))
  mlp:updateParameters(0.01)
end

x = torch.Tensor(2)
x[1] = 0.5; x[2] = 0.5; print(mlp:forward(x))
x[1] = 0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = 0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))  

local model = nn.Sequential()
local inputs = 2; local outputs = 1; local HUs = 20;
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, outputs))

local criterion = nn.MSECriterion()

local batchSize = 128
local batchInputs = torch.Tensor(batchSize, inputs)
local batchLabels = torch.DoubleTensor(batchSize)

for i = 1, batchSize do
  local input = torch.randn(2)
  local label = 1
  if input[1] * input[2] > 0 then
    label = -1
  end
  batchInputs[i]:copy(input)
  batchLabels[i] = label
end

require 'optim'

local optimState = {learningRate = 0.01}
local params, gradParams = model:getParameters()

for epoch = 1, 50 do
  local function feval(params)
    gradParams:zero()
    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    local dloss_doutput = criterion:backward(outputs, batchLabels)
    model:backward(batchInputs, dloss_doutput)
    
    return loss, gradParams
  end
  optim.sgd(feval, params, optimState)
end

x = torch.Tensor(2)
x[1] = 0.5; x[2] = 0.5; print(mlp:forward(x))
x[1] = 0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = 0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))  

local x = torch.Tensor({
{0.5, 0.5},
{0.5, -0.5},
{-0.5, 0.5},
{-0.5, -0.5}
})

print(model:forward(x))




