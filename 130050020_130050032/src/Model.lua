local Model = torch.class("Model")

function  Model:__init( )
	-- body
	-- print ("called")
	self.Layers = {}
	self.count = 0
	return
end

function Model:forward( input )
	-- body
	layerInput = input
	for i =1, self.count do
		layerInput = self.Layers[i]:forward(layerInput)
	end
	return layerInput
end

function  Model:backward( input,gradOutput )
	-- body
	layerOutput = gradOutput
	for i=1,self.count-1 do
		layerOutput = self.Layers[self.count - i+1]:backward(self.Layers[self.count - i].output,layerOutput)
	end
	layerOutput = self.Layers[1]:backward(input,layerOutput)
	return layerOutput
end

function  Model:dispGradParam()
	-- body
	for i=1,self.count do
		self.Layers[self.count - i+1]:printParam()
	end
end

function Model:clearGradParam()
	-- body
	for i=1,self.count do
		self.Layers[self.count - i+1]:clearParam()
	end
end

function Model:addLayer(object)
	-- body
	table.insert(self.Layers,object)
	self.count = self.count + 1
end

function Model:addWB(index, W,B)
	-- print(W:size(1))
	-- print(W:size(2))
	-- print(B:size(1))
	self.Layers[index]:setWB(W,B)	
end
