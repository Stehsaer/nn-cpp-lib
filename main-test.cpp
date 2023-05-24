#include "nn-cpp-lib.h"

#include <iostream>
#include <math.h>
#include <format>

//== helper functions

void print_vector(const nn::vector& v)
{
	std::cout << v.format_str() << std::endl;
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

class mnist_network :public nn::base_network<nn::matrix, nn::vector>
{
public:
	nn::input_layer::vector_input input = nn::input_layer::vector_input(28 * 28);
	nn::hidden_layer::linear_layer linear = nn::hidden_layer::linear_layer(48, 28 * 28);
	nn::hidden_layer::linear_layer linear2 = nn::hidden_layer::linear_layer(10, 48);
	nn::optimizer::softmax_optimizer softmax = nn::optimizer::softmax_optimizer(10);

	const nn::activate_func* func = new nn::leaky_relu_func();

	void feed_data(const nn::matrix& data)
	{
		input.push_input(data.to_vector());
	}

	nn::vector get_output()
	{
		return softmax.get_output();
	}

	void forward_and_grad(const nn::vector& target)
	{
		softmax.push_target(target);

		linear.forward(&input, func);
		linear2.forward(&linear, func);
		softmax.forward_and_grad(&linear2);
	}

	void backward()
	{
		linear2.backward(&softmax);
		linear.backward(&linear2);
	}

	void forward()
	{
		linear.forward(&input, func);
		linear2.forward(&linear, func);
		softmax.forward(&linear2);
	}

	void update_weights()
	{
		linear.update_weights(&input, func, learning_rate);
		linear2.update_weights(&linear, func, learning_rate);
	}

	float get_loss()
	{
		return softmax.get_loss();
	}

	void init_weights(float min, float max)
	{
		linear.rand_weights(min, max);
		linear2.rand_weights(min, max);
	}
};

int main()
{
	nn::mnist_dataset set;
	set.add_source(get_line("data-path"), get_line("label-path"));

	mnist_network network;
	network.init_weights(0.1, 0.9);

	int count = 0;

	for (auto& item : set.set)
	{
		network.feed_data(item->get_data());
		network.forward_and_grad(item->get_target());
		network.backward();
		network.update_weights();

		if (count % 100 == 0)
		{
			printf("index: %d,\tloss: %.4f\n", count, network.get_loss());
		}
		
		count++;
	}

	int correct = 0, wrong = 0;
	for (auto& item : set.set)
	{
		network.feed_data(item->get_data());
		network.forward();

		auto result = network.get_output();
		auto max = result.max();

		if (max.index == item->get_label())
			correct++;
		else
			wrong++;
	}

	printf("verify: correct=%d, wrong=%d", correct, wrong);
	
	return 0;
}