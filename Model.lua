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

end

function  Model::dispGradParam( )
	-- body
end

function Model::clearGradParam()

end

function Model::addLayer( Layer class object )
	-- body
	self.Layers[self.count] = object
	self.count++
end