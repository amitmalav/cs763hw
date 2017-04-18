local ReLU = torch.class("ReLU")
function ReLU:__init()
	self.output = nil
	self.gradInput = torch.zeros(10):float()


-- L_i = -log(e^yi/sum_j(e^yj)) + lambda*||W||^2
function ReLU:forward(op, ti, model, lambda)
	op = op:exp()
	op = op/(op:sum())
	local lossL2 = model.W:clone()
	lossL2 = lossL2:cmul(lossL2):sum()
	self.output = -math.log(op[ti]) + lambda * lossL2
	return self.output
end

function ReLU:backward(op, ti)
    self.gradInput:fill(0)
    self.gradInput[ti] = 1
    self.gradInput = self.gradInput - op
    return self.gradInput
end