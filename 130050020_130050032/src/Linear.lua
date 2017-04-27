local Linear = 	torch.class("Linear")

-- init function
function Linear:__init(inp_neuron,out_neuron)
	self.output = nil
	self.W = torch.zeros(out_neuron,inp_neuron)
	self.B = torch.zeros(out_neuron,1)
	self.gradW = torch.zeros(out_neuron,inp_neuron)
	self.gradB = torch.zeros(out_neuron,1)
	self.gradInput = nil
end

-- calculates W* input + B and returns
function Linear:forward(input)
	-- print(input:size(1))
	-- print(input:size(2))
	-- -- input = input:t()
	-- print(self.B:size(1))
	-- print(input:size(2))
	kay = self.W * input
	self.output = kay + torch.repeatTensor(self.B,1,input:size(2)):resize(input:size(2),self.B:size(1)):t()
	return self.output
end

-- gradinput = dL/dn gradOutput = dL/dm
-- gradW = dL/dW gradB = dL/dB
function Linear:backward(input,gradOutput)
	-- print("dfdfgdf")
	self.gradInput = gradOutput * self.W
	gradO = gradOutput
	-- print(gradO)
	if(torch.isTensor(gradO)) then gradO = gradO:t() end
	-- print(input:size(1))
	-- print(input:size(2))
	self.gradW = gradO * input:t()
	if(torch.isTensor(gradO)) then self.gradB = gradOutput:sum(1):t() else self.gradB = gradOutput end
	return self.gradInput
end

function  Linear:printParam()
	-- body
	print(unpack{self.gradW,self.gradB})
end

function  Linear:clearParam()
	-- body
	-- self.gradW = torch.tensor(out_neuron,inp_neuron):zero()
	-- self.gradB = torch.tensor(out_neuron,1):zero()
end

function Linear:setWB(Ww,Bb)
	self.W = Ww
	self.B = Bb
end