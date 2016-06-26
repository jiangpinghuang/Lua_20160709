-- SemEval-2015 Task1 PIT

local PIT = torch.class('PIT')

local function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end 

local function sentSplit(sent, sep)
  local tokens = {}  
  while (true) do
    local pos = string.find(sent, sep)
    if (not pos) then
      tokens[#tokens + 1] = sent
      break
    end
    local token = string.sub(sent, 1, pos - 1)
    tokens[#tokens + 1] = token
    sent = string.sub(sent, pos + 1, #sent)
  end  
  return tokens
end

function PIT:readEmb(vocab, emb)

end

function PIT:readSent(path, vocab)
  local sents = {}
  local file = io.open(path, 'r')
  local line  
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = strSplit(line, " ")
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sents[#sents + 1] = sent
  end  
  file:close()
  return sents
end

function PIT:readData(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsent = PIT.readSent(dir .. 'l.toks', vocab)
  dataset.rsent = PIT.readSent(dir .. 'r.toks', vocab)
  dataset.size  = #dataset.lsent
  local id = torch.DiskFile(dir .. 'id.txt')
  local sim = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)  
  for i = 1, dataset.size do
    dataset.ids[i] = id:readInt() 
    dataset.labels[i] = sim:readDouble()
  end  
  id:close()
  sim:close()  
  return dataset
end

function PIT:__init(config)
  self.layer            = config.layer            or 1
  self.dim              = config.dim              or 300
  self.learningRate     = config.learningRate     or 0.01
  self.epoch            = config.epoch            or 50
  self.batchSize        = config.batchSize        or 25
end

local function config()
  local layer = 1
  local dim = 300
  local learningRate = 0.01
  local epoch = 50
  local batchSize = 25
end

function PIT:train(train)

end

function PIT:trainDev(train, dev)

end

function PIT:predict(model, test)

end

function PIT:save(model)
  local config = {
    layer         = self.layer,
    dim           = self.dim,
    learningRate  = self.learningRate,
    epoch         = self.epoch,
    batchSize     = self.batchSize,
  }
  torch.save(model, {
    params = self.params,
    config = config,
  })
end

local function main()
  local train = '/home/hjp/Workshop/Model/coling/pit/train.txt'
  local dev = '/home/hjp/Workshop/Model/coling/pit/dev.txt'
  local test = '/home/hjp/Workshop/Model/coling/pit/test.txt'
  local vocab = '/home/hjp/Workshop/Model/coling/pit/vocab.txt'
  local vector = '/home/hjp/Workshop/Model/coling/pit/embedding.txt'
  local model = '/home/hjp/Workshop/Model/coling/pit/model'
  local result = '/home/hjp/Workshop/Model/coling/pit/result.txt'
  
  header('demo')
  sentSplit("A hello world in Lua !", " ")
end

main()