-- Timer

-- This class is able to measure time (in seconds) elapsed in a particular.
timer = torch.Timer()
x = 0
for i = 1, 1000000000 do
  x = x + math.sin(x)
end
print('Time elapsed for 1,000,000,000 sin: ' .. timer:time().real .. ' seconds.')