local Linear = 	torch.class("Linear")

-- init function
function Linear::__init(inp_neuron,out_neuron)
	self.output = nil
	self.W = torch.tensor(out_neuron,inp_neuron):zero()
	self.B = torch.tensor(out_neuron,1):zero()
	self.gradW = torch.tensor(out_neuron,inp_neuron):zero()
	self.gradB = torch.tensor(out_neuron,1):zero()
	self.gradInput = nil
end

-- calculates W* input + B and returns
function Linear:forward(input)
	self.output = self.W * input + self.B
	return self.output
end

-- gradinput = dL/dn gradOutput = dL/dm
-- gradW = dL/dW gradB = dL/dB
function Linear:backward(input,gradOutput)
	gradInput = gradOutput * self.W
	gradW = gradOutput * input
	gradB = gradOutput
end