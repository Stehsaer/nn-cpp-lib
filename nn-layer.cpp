#include "nn-layer.h"

nn::input_layer::vector_input::vector_input(size_t size)
{
	input_size = size;
	input = vector(size);
}

void nn::input_layer::vector_input::input_data(vector& data)
{
	if (data.size() != input_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	input = data;
}

const nn::vector& nn::input_layer::vector_input::get_input()
{
	return input;
}

nn::optimizer::mse_optimizer::mse_optimizer(size_t size)
{
	optimizer_size = size;
	output = vector(size);
	gradient = vector(size);
}

void nn::optimizer::mse_optimizer::forward_and_grad(hidden_layer::linear_layer& layer)
{
	// check element count
	if(layer.get_size()!=optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	auto& prev_input = layer.get_value();
	
	// loss
	output.do_foreach([this, &prev_input](size_t idx) {
		output[idx] = prev_input[idx] - target[idx];
		output[idx] *= output[idx];
		});
}

const nn::vector& nn::optimizer::vector_optimizer::get_gradient()
{
	return gradient;
}

const nn::vector& nn::optimizer::vector_optimizer::get_output()
{
	return output;
}

void nn::optimizer::vector_optimizer::push_target(vector& target)
{
	if(target.size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	this->target = target;
}
