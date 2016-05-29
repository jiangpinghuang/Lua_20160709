--[[
Lua basis function.
--]]
-- Obtain system date information. --
print(os.date('%Y%m%d%H%M%S'))
-- Obtain system current time. --
local timer = torch.Timer()
local time1 = timer:time().real
print(time1)