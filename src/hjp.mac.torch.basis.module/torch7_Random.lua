-- Random

-- Generator handling
x = torch.manualSeed(0)
y = torch.random()
print(x)
print(y)

gen = torch.Generator()
x = torch.manualSeed(gen, 0)
y = torch.random(gen)
print(x)
print(y)

print(torch.random(gen))
print(torch.random())

-- Seed Handling
x = torch.manualSeed(123)
print(x)
print(torch.uniform())
print(torch.uniform())
torch.manualSeed(123)
print(torch.uniform())
print(torch.uniform())
torch.manualSeed(torch.initialSeed())
print(torch.uniform())
print(torch.uniform())
print(torch.uniform())

torch.manualSeed(456)
print(torch.uniform())
print(torch.uniform())
s = torch.getRNGState()
print(torch.uniform())
print(torch.uniform())
print(torch.uniform())
torch.setRNGState(s)
print(torch.uniform())
print(torch.uniform())
print(torch.uniform())
print(torch.random())
print(torch.normal())
print(torch.bernoulli())