local ReLU = torch.class("ReLU")
function ReLU:__init()
	self.output = nil
	self.gradInput = nil

--output = max(0, input)
function ReLU:forward(input)
	self.output = torch.max(0, input)
	return self.output
end

--gradLoss = dm/dn
--gradOutput = dL/dm
--gradinput = dL/dn = dL/dm * dm/dn

function ReLU:backward(input, gradOutput)
	loss = 1
	if input < 0 then
		loss = 0
	self.gradInput = gradOutput * loss
	return self.gradInput
end

function  ReLU:printParam()
	-- body
	print(self.gradOutput)
end

function  ReLU:clearParam()
	-- body
	-- self.gradW = torch.tensor(out_neuron,inp_neuron):zero()
	-- self.gradB = torch.tensor(out_neuron,1):zero()

end