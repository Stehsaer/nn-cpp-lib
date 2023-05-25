// FILENAME: nn-example.h
// THIS HEADER IS OPTIONAL!
// Provide examples for some typical network structures, for reference
// use #define ENABLE_NN_EXAMPLES

#ifdef ENABLE_NN_EXAMPLES

#include "nn-layer.h"

namespace nn::examples
{
	/*
	TYPICAL TRAINING & VERIFYING PROCESS:

	==[TRAIN]==

	load_dataset();
	for_each_data:
	{
		feed_data([data]);
		forward_and_grad([target]);
		backward();
		update_weights();
	}

	==[VERIFY]==

	load_dataset();
	for_each_data:
	{
		feed_data([data]);
		[compare: get_output()[.max(), optional], [label]]
		count();
	}
	print(...);
	
	*/

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

		nn::vector& get_output()
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
}

#endif