-- SemEval-2015 Task1 PIT

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

PIT = {}

include('Dict.lua')

local function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end 

function PIT.sentSplit(sent, sep)
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

function PIT.readEmb(voc, emb)
  local vocab = PIT.Dict(voc)
  local embed = torch.load(emb)
  return vocab, embed
end

function PIT.readSent(path, vocab)
  local sents = {}
  local file = io.open(path, 'r')
  local line  
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = PIT.sentSplit(line, " ")
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

function PIT.readData(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsent = PIT.readSent(dir .. 'ls.toks', vocab)
  dataset.rsent = PIT.readSent(dir .. 'rs.toks', vocab)
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

function PIT.train(train)

end

function PIT.trainDev(train, dev)

end

function PIT.predict(model, test)

end

function PIT.save(model)
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
  header('Loading vectors...')
  local vocDir = '/home/hjp/Workshop/Model/coling/pit/vocabs.txt'
  local vocab = PIT.Dict(vocDir)
  local eVocDir = '/home/hjp/Workshop/Model/coling/vec/twitter.vocab'
  local eDimDir = '/home/hjp/Workshop/Model/coling/vec/twitter.th'
  local eVoc, eVec = PIT.readEmb(eVocDir, eDimDir)
  local dimSize = eVec:size(2)
  
  local vecs = torch.Tensor(vocab.size, dimSize)
  for i = 1, vocab.size do
    local w = vocab:token(i)
    if eVoc:contains(w) then
      vecs[i] = eVec[eVoc:index(w)]
    else
      vecs[i]:uniform(-0.05, 0.05)
    end
  end

  eVoc, eVec = nil, nil
  collectgarbage()

  header('Loading datasets...')
  local trainDir = '/home/hjp/Workshop/Model/coling/pit/train/'
  local devDir = '/home/hjp/Workshop/Model/coling/pit/dev/'
  local testDir = '/home/hjp/Workshop/Model/coling/pit/test/'
  local trainSet = PIT.readData(trainDir,vocab)
  local devSet = PIT.readData(devDir,vocab)
  local testSet = PIT.readData(testDir,vocab)
  print('train size: ' .. trainSet.size)
  print('dev size: ' .. devSet.size)
  print('test size: ' .. testSet.size)
  
  local vocab = '/home/hjp/Workshop/Model/coling/pit/vocab.txt'
  local vector = '/home/hjp/Workshop/Model/coling/pit/embedding.txt'
  local model = '/home/hjp/Workshop/Model/coling/pit/model'
  local result = '/home/hjp/Workshop/Model/coling/pit/result.txt'
  local evoc = '/home/hjp/Workshop'
  
  header('demo')
end

main()