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
cmd:option('-o', 'output.bin', '/path/to/output.bin')
cmd:option('-ow', 'gradWeight.bin', '/path/to/gradWeight.bin')
cmd:option('-ob', 'gradB.bin', '/path/to/gradB.bin')
cmd:option('-og', 'gradInput.bin', '/path/to/gradInput.bin')

params = cmd:parse(arg)

-- print(params.config)
conf = io.open(params.config)
count_layer = tonumber(conf:read())
-- print (count_layer)

myModel = Model()

linearList = {}
linearCount = 0
for i = 1,count_layer do
	layerType = conf:read()
	-- print(layerType)
	layerData = layerType:gmatch("%w+")
	t = {}
	for w in layerData do table.insert(t,w) end
	if t[1] == "linear" then
		table.insert(linearList,i)
		linearCount = linearCount + 1
		myModel:addLayer(Linear(tonumber(t[2]),tonumber(t[3])))
	else
		myModel:addLayer(ReLU())
	end
end
w_file = conf:read()
-- print(w_file)
w_inp = torch.load(w_file)
b_inp = torch.load(conf:read())
-- print(w_inp)
-- print(b_inp)
for i = 1,linearCount do
	myModel:addWB(linearList[i],w_inp[i],b_inp[i])
end
input = torch.load(params.i)
gradOuput = torch.load(params.ig)
input:resize(input:size(1), input:size(2) * input:size(3) * input:size(4))
input = input:t()
-- gradOuput:resize(gradOuput:size(1), gradOuput:size(2) * gradOuput:size(3) * input:size(4))
-- print(input:size(1))
-- print("blahnal")
out = myModel:forward(input)

torch.save(params.o, out)

gradInp = myModel:backward(input,gradOuput)

gradW = {}
gradB = {}

for i=1,linearCount do
	table.insert(gradW,myModel.Layers[linearList[i]].gradW)
	table.insert(gradB,myModel.Layers[linearList[i]].gradB)
end

torch.save(params.ow,gradW)
torch.save(params.ob,gradB)
torch.save(params.og,gradInp)

-- while true do = 
-- 	line = conf:read()
-- 	if line == nil then break end
