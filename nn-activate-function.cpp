#include "nn-activate-function.h"

#include <math.h>

float nn::relu_func::forward(float x) const
{
	return x > 0 ? x : 0.0f;
}

float nn::relu_func::backward(float x) const
{
	return x > 0 ? 1.0f : 0.0f;
}

float nn::leaky_relu_func::forward(float x) const
{
	return x > 0 ? x : 0.1f * x;
}

float nn::leaky_relu_func::backward(float x) const
{
	return x > 0 ? 1.0f : 0.1f;
}

float nn::sigmoid_func::forward(float x) const
{
	return 1.0f / (1.0f + expf(-x));
}

float nn::sigmoid_func::backward(float x) const
{
	return x * (1.0f - x);
}