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
		layerInput = later.forward(layerInput)
	end
	return layerInput
end

function  Model::backward( input,gradOutput )
	-- body
	layerOutput = gradOutput
	for i=1,self.count do
		layerOutput = self.Layers[self.count - i].backward(self.Layers[self.count - i - 1].output,layerOutput)
	end
	layerOutput = self.Layers[0].backward(input,layerOutput)
end

function  Model::dispGradParam( )
	-- body
	for i=1,self.count do
		print self.Layers[self.count - i].W, self.Layers[self.count - i].B
end

function Model::clearGradParam()
	for i=1,self.count do
		print(i)
	end
end

function Model::addLayer( Layer class object )
	-- body
	self.Layers[self.count] = object
	self.count = self.count + 1
end