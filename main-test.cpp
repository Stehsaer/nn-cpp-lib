#include "nn-cpp-lib.h"

#include <iostream>
#include <math.h>
#include <format>

#define ENABLE_NN_EXAMPLES
#include "nn-example.h"

//== helper functions

void print_vector(const nn::vector& v)
{
	std::cout << v << std::endl;
}

template<typename T>
void get_input_number(std::string prefix, T& num)
{
	std::cout << prefix << "=";
	std::cin >> num;
}

std::string get_line(std::string prefix)
{
	std::string str;
	std::cout << prefix << "=";
	std::getline(std::cin, str);
	return str;
}

void mnist_train_test()
{
	nn::mnist_dataset set;
	set.add_source(get_line("data-path"), get_line("label-path"));

	nn::examples::mnist_network network;
	network.init_weights(0.1, 0.9);

	int count = 0;

	for (auto& item : set.set)
	{
		network.feed_data(item->get_data());
		network.forward_and_grad(item->get_target());
		network.backward();
		network.update_weights();

		count++;
	}

	int correct = 0, wrong = 0;
	for (auto& item : set.set)
	{
		network.feed_data(item->get_data());
		network.forward();

		auto& result = network.get_output();
		auto max = result.max();

		if (max.index == item->get_label())
			correct++;
		else
			wrong++;
	}

	printf("verify: correct=%d, wrong=%d", correct, wrong);
}

int main()
{
	nn::mnist_dataset set;
	set.add_source(get_line("data-path"), get_line("label-path"));

	nn::input_layer::matrix_input input(28, 28);
	nn::hidden_layer::conv2_layer conv1(28, 28, 1, 3, 1, 1);
	nn::hidden_layer::relu_layer relu(28, 28, 1);

	input.push_input(set.set[1]->get_data());

	conv1.rand_weights(0.1, 0.9);

	conv1.forward(&input);
	relu.forward(&conv1);
	
	relu.get_map(0).export_image("conv.jpg", 100);
	
	return 0;
}