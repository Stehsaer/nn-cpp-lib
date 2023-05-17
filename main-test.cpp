#include "nn-cpp-lib.h"

#include <iostream>

void print_vector(const nn::vector& v)
{
	std::cout << v.format_str() << std::endl;
}

int main()
{
	nn::input_layer::vector_input input(5);
	nn::hidden_layer::linear_layer layer(5, 5);
	nn::optimizer::softmax_optimizer opt(5);

	layer.rand_weights(0.1, 0.9);

	nn::vector target(5);
	nn::math::rand_vector(target, 0.1, 0.9);

	input.input_data(target);
	print_vector(input.get_input());

	nn::activate_func* func = new nn::sigmoid_func();
	layer.forward(&input, func);
	print_vector(layer.get_value());

	opt.push_target(target);
	opt.forward_and_grad(layer);
	print_vector(opt.get_output());
	print_vector(opt.get_gradient());

	layer.backward(&opt);
	print_vector(layer.get_gradient());

	return 0;
}