local ReLU = torch.class("ReLU")
function ReLU:__init()
	self.output = nil
	self.gradInput = nil
end
--output = max(0, input)
function ReLU:forward(input)
	self.output = torch.max(0, input)
	return self.output
end

function ReLU:backward(input, gradOutput)
	loss = torch.zeros(input:size(1), input:size(2))
	for i = 1, input:size(1) do
		for j = 1, input:size(2) do
			loss[i][j] = 0
			if input[i][j] > 0  then
				loss[i][j] = 1
			end
		end
	end
	self.gradInput = torch.dot(gradOutput, loss)
	return self.gradInput
end