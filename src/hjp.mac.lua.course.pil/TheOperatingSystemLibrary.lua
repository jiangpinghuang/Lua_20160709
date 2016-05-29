print(os.time{year=1970, month=1, day=2, hour=0})

print(os.date("a %A in %B"))
print(os.date("%x",101000490))

local x = os.clock()
local s = 0
for i = 1, 100000 do s = s + i end
print(string.format("elapsed time: %.2f\n", os.clock() - x))

print(os.getenv("HOME"))