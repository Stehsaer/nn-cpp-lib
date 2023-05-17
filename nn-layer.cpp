#include "nn-layer.h"

#include <math.h>

nn::input_layer::vector_input::vector_input(size_t size)
{
	input_size = size;
	input = vector(size);
}

void nn::input_layer::vector_input::input_data(const vector & data)
{
	if (data.size() != input_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	input = data;
}

nn::vector& nn::input_layer::vector_input::get_input()
{
	return input;
}

size_t nn::input_layer::vector_input::get_size()
{
	return input_size;
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
	if (layer.get_size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	auto& prev_input = layer.get_value();

	// forward
	output = prev_input;

	// calculate gradient
	gradient.for_each([this, &prev_input](size_t idx, float& num)
		{
			num = target[idx] - output[idx];
		});
}

nn::vector& nn::optimizer::vector_optimizer::get_gradient()
{
	return gradient;
}

nn::vector& nn::optimizer::vector_optimizer::get_output()
{
	return output;
}

size_t nn::optimizer::vector_optimizer::get_size()
{
	return optimizer_size;
}

void nn::optimizer::vector_optimizer::push_target(vector& target)
{
	if (target.size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	this->target = target;
}

nn::optimizer::softmax_optimizer::softmax_optimizer(size_t size)
{
	optimizer_size = size;
	output = vector(size);
	gradient = vector(size);
}

void nn::optimizer::softmax_optimizer::forward_and_grad(hidden_layer::linear_layer& layer)
{
	// check element count
	if (layer.get_size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	auto& prev_input = layer.get_value();
	output = prev_input;
	auto max = prev_input.max();

	// do softmax
	output.for_each([&prev_input, &max](size_t idx, float& num)
		{
			num = exp(prev_input[idx] - max.result);
		});

	output /= output.sum();

	// calculate gradient
	gradient.for_each([this](size_t idx, float& num)
		{
			num = target[idx] - output[idx];
		});
}

nn::hidden_layer::linear_layer::linear_layer(size_t size, size_t weight_size)
{
	num_weights = weight_size;
	num_neurons = size;

	weights.resize(size, vector(weight_size));
	value = vector(size); value.fill(0.0f);
	gradient = vector(size); value.fill(0.0f);

	bias = 0.0f;
}

void nn::hidden_layer::linear_layer::forward(input_layer::vector_input* prev, const activate_func* func)
{
	if (prev->get_size() != num_neurons)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	value.for_each([this, &prev, func](size_t idx, float& num)
		{
			num = func->forward(nn::vector::dot(prev->get_input(), weights[idx]) + bias);
		});
}

void nn::hidden_layer::linear_layer::forward(linear_layer* prev, activate_func* func)
{
	if (prev->get_size() != num_neurons)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	value.for_each([this, &prev, func](size_t idx, float& num)
		{
			num = func->forward(nn::vector::dot(prev->get_value(), weights[idx]) + bias);
		});
}

void nn::hidden_layer::linear_layer::backward(optimizer::vector_optimizer* optimizer)
{
	if(optimizer->get_size() != num_neurons)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	gradient = optimizer->get_gradient();
}

nn::vector& nn::hidden_layer::linear_layer::get_value()
{
	return value;
}

nn::vector& nn::hidden_layer::linear_layer::get_gradient()
{
	return gradient;
}

nn::vector& nn::hidden_layer::linear_layer::get_weight(size_t index)
{
	return weights[index];
}

size_t nn::hidden_layer::linear_layer::get_size()
{
	return num_neurons;
}

void nn::hidden_layer::linear_layer::rand_weights(float min, float max)
{
	for (auto& weight : weights)
		nn::math::rand_vector(weight, min, max);
}
