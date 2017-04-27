local Criterion = torch.class("Criterion")

function Criterion:__init()
	-- body
end

function Criterion:forward( input, target )
	avg_loss = 0
	for i = 1, input:size(1) do
		input[i] = input[i]:exp()
		input[i] = input[i]/(input[i]:sum())
		tmp = torch.zeros(input:size(1), input:size(2))
		avg_loss = avg_loss - math.log(input[i][target[i]])
	end
	avg_loss = avg_loss/input:size(1)
	return avg_loss
end

function Criterion:backward( input, target )
	-- body
	gradInput = torch.zeros(input:size(1), input:size(2))
	for i = 1, input:size(1) do
    	gradInput[i][target[i]] = 1
    	gradInput[i] = gradInput[i] - input
	end
	return gradInput

end