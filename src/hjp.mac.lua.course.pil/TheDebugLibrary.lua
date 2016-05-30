-- Chap 24 in PIL.

function traceback ()
  for level = 1, math.huge do
    local info = debug.getinfo(levle, "Sl")
    if not info then break end
    if info.what == "C" then 
      print(level, "C function")
    else
      print(string.format("[%s]:%d", info.short_src, info.currentline))
    end
  end
end

function foo (a, b)
  local x
  do local c = a - b end
  local a = 1
  while true do
    local name, value = debug.getlocal(1, a)
    if not name then break end
    print(name, value)
    a = a + 1
  end
end

foo(10, 20)

function getvarvalue (name, level)
  local value
  local found = false
  
  level = (level or 1) + 1
  
  for i = 1, math.huge do
    local n, v = debug.getlocal(level, i)
    if not n then break end
    if n == name then
      value = v
      found = true
    end
  end
  
  if found then return value end
  
  local func = debug.getinfo(level, "f").func
  for i = 1, math.huge do 
    local n, v = debug.getupvalue(func, i)
    if not n then break end
    if n == name then return v end
  end
  
  local env = getvarvalue("_ENV", level)
  return env[name]
end

function trace (event, line)
  local s = debug.getinfo(2).short_src
  print(s .. ":" .. line)
end

debug.sethook(trace, "1")

function debug1 ()
  while true do
    io.write("debug> ")
    local line = io.read()
    if line == "cont" then break end
    assert(load(line))()
  end
end

local Counters = {}
local Names = {}

local function hook ()
  local f = debug.getinfo(2, "f").func
  local count = Counters[f]
  if count == nil then
    Counters[f] = 1
    Names[f] = debug.getinfo(2, "Sn")
  else
    Counters[f] = count + 1
  end
end

function getname (func)
  local n = Names[func]
  if n.what == "C" then
    return n.name
  end
  local lc = string.format("[%s]:%d", n.short_src, n.linedefined)
  if n.what ~= "main" and n.namewhat ~= "" then
    return string.format("%s (%s)", lc, n.name)
  else
    return lc
  end
end
