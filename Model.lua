local Model = torch.class("Model")

function  Model::__init( )
	-- body
	self.Layers = {}
	self.count = 0
end

function Model::forward( input )
	-- body
	layerInput = input
	for layer in self.Layers do
		layerInput = layer.forward(layerInput)
	end
	return layerInput
end

function  Model::backward( input,gradOutput )
	-- body
	layerOutput = gradOutput
	for i=1,self.count-1 do
		layerOutput = self.Layers[self.count - i].backward(self.Layers[self.count - i - 1].output,layerOutput)
	end
	layerOutput = self.Layers[0].backward(input,layerOutput)
end

function  Model::dispGradParam()
	-- body
	for i=1,self.count do
		self.Layers[self.count - i].printParam()
	end
end

function Model::clearGradParam()
	-- body
	for i=1,self.count do
		self.Layers[self.count - i].clearParam()
	end
end

function Model::addLayer(object)
	-- body
	self.Layers[self.count] = object
	self.count = self.count + 1
end