path = "/home/hjp/Downloads/abc.txt"

-- Opens a file in read
local file = io.open(path, "r")
print(file:read())
for line in io.lines(path) do
  print(line)
end
file:close()

-- Open a file in append
local file1 = io.open(path,"a")
file1:write("\nend line")
file1:close()