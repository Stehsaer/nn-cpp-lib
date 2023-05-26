#include "nn-layer.h"

#include <math.h>

nn::input_layer::vector_input::vector_input(size_t size)
{
	input_size = size;
	input = vector(size);
}

void nn::input_layer::vector_input::push_input(const vector& data)
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
	target = vector(size);
}

void nn::optimizer::mse_optimizer::forward_and_grad(hidden_layer::linear_layer* layer)
{
	// check element count
	if (layer->get_size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	auto& prev_input = layer->get_value();
	loss = 0.0f;

	// forward
	output = prev_input;

	// calculate gradient
	gradient.for_each([this, &prev_input](size_t idx, float& num)
		{
			num = target[idx] - output[idx];
			loss += num * num;
		});
}

void nn::optimizer::mse_optimizer::forward(hidden_layer::linear_layer* layer)
{
	output = layer->get_value();
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

float nn::optimizer::vector_optimizer::get_loss()
{
	return loss;
}

void nn::optimizer::vector_optimizer::push_target(const vector& target)
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
	target = vector(size);
}

void nn::optimizer::softmax_optimizer::forward_and_grad(hidden_layer::linear_layer* layer)
{
	// check element count
	if (layer->get_size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	forward(layer);

	// calculate gradient
	gradient.for_each([this](size_t idx, float& num)
		{
			num = target[idx] - output[idx];
			loss += -target[idx] * log(output[idx]);
		});
}

void nn::optimizer::softmax_optimizer::forward(hidden_layer::linear_layer* layer)
{
	if (layer->get_size() != optimizer_size)
		throw nn::logic_exception("vector size mismatch!", __FUNCTION__, __LINE__);

	auto& prev_input = layer->get_value();
	output = prev_input;
	loss = 0.0f;

	auto max = prev_input.max();

	// do softmax
	output.for_each([&prev_input, &max](size_t idx, float& num)
		{
			num = exp(prev_input[idx] - max.result);
		});

	output /= output.sum();
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
	if (prev->get_size() != num_weights)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	value.for_each([this, prev, func](size_t idx, float& num)
		{
			auto& input = prev->get_input();
			num = func->forward(nn::vector::dot(input, weights[idx]) + bias);
		});
}

void nn::hidden_layer::linear_layer::forward(linear_layer* prev, const activate_func* func)
{
	if (prev->get_size() != num_weights)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	value.for_each([this, &prev, func](size_t idx, float& num)
		{
			num = func->forward(nn::vector::dot(prev->get_value(), weights[idx]) + bias);
		});
}

void nn::hidden_layer::linear_layer::forward(conv2_linear_adapter_layer* prev, const activate_func* func)
{
	if(prev->size != num_weights)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	value.for_each([this, &prev, func](size_t idx, float& num)
		{
			num = func->forward(nn::vector::dot(prev->get_value(), weights[idx]) + bias);
		});
}

void nn::hidden_layer::linear_layer::backward(optimizer::vector_optimizer* optimizer)
{
	if (optimizer->get_size() != num_neurons)
		throw logic_exception("size mismatch!", __FUNCTION__, __LINE__);

	gradient = optimizer->get_gradient();
}

void nn::hidden_layer::linear_layer::backward(linear_layer* last)
{
	gradient.for_each([this, last](size_t idx, float& num)
		{
			num = 0.0f;

			last->get_gradient().for_each([&num, idx, last](size_t idx2, float& num2)
				{
					num += last->get_weight(idx2)[idx] * num2;
				});
		});
}

void nn::hidden_layer::linear_layer::update_weights(input_layer::vector_input* prev, const activate_func* func, float learning_rate)
{
	for (size_t i = 0; i < num_neurons; i++)
	{
		bias += learning_rate * func->backward(bias) * gradient[i]; // update bias
		float coeff = learning_rate * func->backward(value[i]) * gradient[i];
		auto& weight = weights[i];

		weight.for_each([coeff, prev](size_t idx, float& num) // update weights
			{
				num += coeff * prev->get_input()[idx];
			});
	}
}

void nn::hidden_layer::linear_layer::update_weights(hidden_layer::linear_layer* prev, const activate_func* func, float learning_rate)
{
	for (size_t i = 0; i < num_neurons; i++)
	{
		bias += learning_rate * func->backward(bias) * gradient[i]; // update bias
		float coeff = learning_rate * func->backward(value[i]) * gradient[i];
		auto& weight = weights[i];

		weight.for_each([coeff, prev](size_t idx, float& num) // update weights
			{
				num += coeff * prev->get_value()[idx];
			});
	}
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

	bias = nn::math::rand_float(min, max);
}

nn::hidden_layer::conv2_layer::conv2_layer(size_t w, size_t h, size_t depth, size_t kernal_size, size_t stride, size_t padding) :
	w(w), h(h), depth(depth), kernal_size(kernal_size), stride(stride), padding(padding)
{
	kernals.resize(depth, nn::matrix(kernal_size, kernal_size));
	maps.resize(depth, nn::matrix(w, h));
	gradients.resize(depth, nn::matrix(w, h));

	bias.resize(depth, 0.0f);
}

void nn::hidden_layer::conv2_layer::forward(input_layer::matrix_input* prev)
{
	// do conv for each kernal
	for (size_t i = 0; i < depth; i++)
	{
		math::conv_2d(maps[i], prev->get_input(), kernals[i], stride, padding);
		maps[i] += bias[i];
	}
}

void nn::hidden_layer::conv2_layer::forward(hidden_layer::conv2_layer* prev)
{
	if (depth != prev->depth)
		throw numeric_exception("conv layer depth mismatch", __FUNCTION__, __LINE__);

	// do conv for each kernal
	for (size_t i = 0; i < depth; i++)
	{
		math::conv_2d(maps[i], prev->get_map(i), kernals[i], stride, padding);
	}
}

void nn::hidden_layer::conv2_layer::forward(hidden_layer::relu_layer* prev)
{
	if (depth != prev->depth)
		throw numeric_exception("relu layer and conv layer depth mismatch", __FUNCTION__, __LINE__);

	// do conv for each kernal
	for (size_t i = 0; i < depth; i++)
	{
		math::conv_2d(maps[i], prev->get_map(i), kernals[i], stride, padding);
	}
}

nn::matrix& nn::hidden_layer::conv2_layer::get_kernal(size_t idx)
{
	return kernals[idx];
}

nn::matrix& nn::hidden_layer::conv2_layer::get_map(size_t idx)
{
	return maps[idx];
}

nn::matrix& nn::hidden_layer::conv2_layer::get_gradient(size_t idx)
{
	return gradients[idx];
}

void nn::hidden_layer::conv2_layer::rand_weights(float min, float max)
{
	for (size_t i = 0; i < depth; i++)
	{
		math::rand_matrix(kernals[i], min, max);
		bias[i] = math::rand_float(min, max);
	}
}

nn::hidden_layer::relu_layer::relu_layer(size_t w, size_t h, size_t depth) :w(w), h(h), depth(depth)
{
	maps.resize(depth, nn::matrix(w, h));
	gradients.resize(depth, nn::matrix(w, h));
}

void nn::hidden_layer::relu_layer::forward(hidden_layer::conv2_layer* prev)
{
	if (depth != prev->depth)
		throw numeric_exception("relu layer and conv layer depth mismatch", __FUNCTION__, __LINE__);

	// do conv for each kernal
	for (size_t i = 0; i < depth; i++)
	{
		maps[i] = prev->get_map(i);
		maps[i].for_each([prev](size_t, size_t, float& num) // do relu
			{
				if (num < 0.0f) num = 0.0f;
			});
	}
}

nn::matrix& nn::hidden_layer::relu_layer::get_map(size_t idx)
{
	return maps[idx];
}

nn::matrix& nn::hidden_layer::relu_layer::get_gradient(size_t idx)
{
	return gradients[idx];
}

nn::input_layer::matrix_input::matrix_input(size_t w, size_t h)
{
	input = matrix(w, h);
	input_width = w;
	input_height = h;
}

void nn::input_layer::matrix_input::push_input(const matrix& data)
{
	input = data;
}

nn::matrix& nn::input_layer::matrix_input::get_input()
{
	return input;
}

size_t nn::input_layer::matrix_input::get_height()
{
	return input_height;
}

size_t nn::input_layer::matrix_input::get_width()
{
	return input_width;
}

nn::hidden_layer::maxpool_layer::maxpool_layer(size_t w, size_t h, size_t depth) :w(w), h(h), depth(depth)
{
	out_maps.resize(depth, matrix(w, h));
	back_gradients.resize(depth, matrix(w * 2, h * 2));
}

void nn::hidden_layer::maxpool_layer::forward(hidden_layer::relu_layer* prev)
{
	if (prev->w != w * 2 || prev->h != h * 2 || prev->depth != depth)
		throw numeric_exception("layer parameters mismatch", __FUNCTION__, __LINE__);

	// down sampling
	for (size_t d = 0; d < depth; d++)
	{
		auto& prev_map = prev->get_map(d);
		auto& map = out_maps[d];
		map.fill(-INFINITY);

		prev_map.for_each([&map](size_t x, size_t y, float& num)
			{
				float& tgt = map.at(x / 2, y / 2);
				if (tgt < num)
					tgt = num;
			});
	}
}

void nn::hidden_layer::maxpool_layer::forward_and_grad(hidden_layer::relu_layer* prev)
{
	forward(prev);

	for (size_t d = 0; d < depth; d++)
	{
		prev->get_map(d).for_each([this, d](size_t x, size_t y, float& num)
			{
				if (num == out_maps[d].at(x / 2, y / 2)) // is the largest one
				{
					back_gradients[d].at(x, y) = 1.0f;
				}
				else // not the largest ones
				{
					back_gradients[d].at(x, y) = 0.0f;
				}
			});
	}
}

nn::matrix& nn::hidden_layer::maxpool_layer::get_map(size_t idx)
{
	return out_maps[idx];
}

nn::matrix& nn::hidden_layer::maxpool_layer::get_gradient(size_t idx)
{
	return back_gradients[idx];
}

nn::hidden_layer::conv2_linear_adapter_layer::conv2_linear_adapter_layer(size_t w, size_t h, size_t depth) :w(w), h(h), depth(depth), size(w*h*depth)
{
	out_vector = vector(size);
	gradients.resize(depth, matrix(w, h));
}

void nn::hidden_layer::conv2_linear_adapter_layer::forward(hidden_layer::maxpool_layer* prev)
{
	// check input
	if (prev->w != w || prev->h != h || prev->depth != depth)
		throw numeric_exception("width, height or depth mismatch", __FUNCTION__, __LINE__);

	// re-arrange matrix to vector
	for (size_t d = 0; d < depth; d++)
	{
		prev->get_map(d).for_each([this, d](size_t x, size_t y, float& num)
			{
				out_vector[d * w * h + y * w + x] = num;
			});
	}
}

nn::matrix& nn::hidden_layer::conv2_linear_adapter_layer::get_gradient(size_t idx)
{
	return gradients[idx];
}

nn::vector& nn::hidden_layer::conv2_linear_adapter_layer::get_value()
{
	return out_vector;
}
