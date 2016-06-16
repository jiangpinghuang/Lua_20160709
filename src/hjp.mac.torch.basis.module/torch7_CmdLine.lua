-- CmdLine
require 'torch'
require 'sys'

cmd = torch.CmdLine

cmd:text('Training a simple network')
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed')
cmd:option('-booloption', false, 'boolean option')
cmd:option('-stroption', 'mystring', 'string option')


params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})
paths.mkdir(params.rundir)

cmd:log(params.rundir .. '/log', params)

cmd:addTime('your project name', '%F %T')
print('Your log message')
-- the code can be run in terminal