require "math"
local Criterion = torch.class("Criterion")

function Criterion:__init()
	-- body
end

function Criterion:forward( input, target )
	avg_loss = 0
	for i = 1, input:size(1) do
		input[i] = input[i]:exp()
		input[i] = input[i]/(input[i]:sum())
		avg_loss = avg_loss - math.log(input[i][target[i][1]])
	end
	avg_loss = avg_loss/input:size(1)
	return avg_loss
end

function Criterion:backward( input, target )
	-- body
	gradInput = torch.zeros(input:size(1), input:size(2))
	for i = 1, input:size(1) do
    	gradInput[i][target[i][1]] = 1
    	-- print("me")
    	-- print(gradInput[i]:size(1))
    	-- print(input:size(1))
    	    	-- print(gradInput[i]:size(3))

    	gradInput[i] = gradInput[i] - input[i]
	end
	return gradInput

end