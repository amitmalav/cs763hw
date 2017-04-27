local cmd = torch.CmdLine()
cmd:text()
cmd:text('checking model')
cmd:text('Options:')
cmd:option('-config', 'modelConfig.txt', '/path/to/modelConfig.txt')
cmd:option('-i', 'input.bin', '/path/to/input.bin')
cmd:option('-ig', 'gradOuput.bin', '/path/to/gradOuput.bin')
cmd:option('-o', 'output.bin', '/path/to/output.bin')
cmd:option('-ow', 'gradWeight.bin', '/path/to/gradWeight.bin')
cmd:option('-ob', 'gradB.bin', '/path/to/gradB.bin')
cmd:option('-og', 'gradInput.bin', '/path/to/gradInput.bin')

params = cmd:parse(arg)

conf = io.open(params.config)
for line in file:read do
	print(line)
end