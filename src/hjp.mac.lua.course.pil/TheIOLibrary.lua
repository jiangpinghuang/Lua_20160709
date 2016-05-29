-- The example of chapter 22 in <<Programing in Lua>>. --
io.write("sin (3) = ", math.sin(3), "\n")
io.write(string.format("sin (3) = %.4f\n", math.sin(3)))
io.write("hello ", "Lua"); io.write(" Hi", "\n")
--t = io.read("a")
--t = string.gsub(t,"([\128-\255=])",function (c)
--      return string.format("=%02X", string.byte(c))
--    end)
--io.write(t)

print(math.huge)

print(io.open("/Users/hjp/Downloads/Lua.txt","r"))