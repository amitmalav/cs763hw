require "src/Model"
require "src/Linear"
require "src/Criterion"
require "src/ReLU"
local cmd = torch.CmdLine()
cmd:text()
cmd:text('checking model')
cmd:text('Options:')
cmd:option('-config', 'modelConfig.txt', '/path/to/modelConfig.txt')
cmd:option('-i', 'input.bin', '/path/to/input.bin')
cmd:option('-ig', 'gradOuput.bin', '/path/to/gradOuput.bin')
cmd:option('-ow', 'gradWeight.bin', '/path/to/gradWeight.bin')
cmd:option('-ob', 'gradB.bin', '/path/to/gradB.bin')
cmd:option('-og', 'gradInput.bin', '/path/to/gradInput.bin')
cmd:option('-t', 'target.bin', ' /path/to/target.bin')


params = cmd:parse(arg)
input = torch.load(params.i)
target = torch.load(params.t)
Crit = Criterion()
loss = Crit:forward(input,target)
print(loss)
gradInput = Crit:backward(input,target)
torch.save(params.og,gradInput)