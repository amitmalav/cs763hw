local Criterion = torch.class("Criterion")

function Criterion::__init()
	-- body
end

function Criterion::forward( input, target )
	-- body
	for i=1,input:size(1) do
		max_input = torch.max(input,i)
	end
	input_exp = input:exp()
	input_exp = input_exp/input_exp:sum()
	self.output = -math.log(input[target])
	return self.output
end

function Criterion::backward( input, target )
	-- body

end