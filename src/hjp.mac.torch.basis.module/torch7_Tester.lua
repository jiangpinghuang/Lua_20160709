-- Tester
local mytest = torch.TestSuite()
local tester = torch.Tester()

function mytest.TestA()
  local a = torch.Tensor{1, 2, 3}
  local b = torch.Tensor{1, 2, 4}
  tester:eq(a, b, "a and b should be equal")
end

function mytest.testB()
  local a = {2, torch.Tensor{1, 2, 2}}
  local b = {2, torch.Tensor{1, 2, 2.001}}
  tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

function mytest.testC()
  local function myfunc()
    return "hello " ..  world
  end
  tester:assertNoError(myfunc, "myfunc shouldn't give an error")
end

tester:add(mytest)
--tester:run()
--
--tester:run()
--tester:run("test1")
--tester:run({"test2", "test3"})

local tester = torch.Tester()
local tests = torch.TestSuite()

function tests.brokenTest()
  -- ...
end

tester:add(tests):disable('brokenTest'):run()

test = torch.TestSuite()
function test.myTest()
  -- ..
end

function test.MyTest()
  -- ..
end
