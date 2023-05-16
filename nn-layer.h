// This header defines different layers of neuron network
// All structures only holds data, without any implementation (forward transmit etc.)

#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn-exception.h"
#include "nn-math.h"
#include "nn-activate-function.h"

namespace nn
{
	//== predefinitions

	namespace input_layer
	{
		struct vector_input;
		struct matrix_input;
		struct tensor_input;
	}

	namespace hidden_layer
	{
		struct linear_layer;
		struct conv_linaer_adapter_layer;
		struct conv_layer;
		struct relu_layer;
		struct maxpool_layer;
	}

	namespace optimizer
	{
		struct vector_optimizer; // base struct for vector-type optimizers
		struct mse_optimizer;
		struct softmax_optimizer;
	}

	//== declaration

	namespace input_layer
	{
		struct vector_input
		{
		private:
			vector input;
			size_t input_size;

		public:
			vector_input(size_t size);

			void input_data(vector& data);
			const vector& get_input();
		};
	}

	namespace hidden_layer
	{
		struct linear_layer
		{
		private:
			vector value, error;
			vector* weights;
			size_t layer_size;

		public:
			linear_layer(size_t size);

			void forward(input_layer::vector_input& prev);
			void forward(linear_layer& prev);

			vector& get_value();
			vector& get_error();
			vector& get_weight(size_t index);
			size_t get_size();
		};
	}

	namespace optimizer
	{
		struct vector_optimizer
		{
		protected:
			vector output, gradient, target;
			size_t optimizer_size = 0;

		public:
			const vector& get_gradient();
			const vector& get_output();
			void push_target(vector& target);
		};

		struct mse_optimizer :vector_optimizer
		{
		public:
			mse_optimizer(size_t size);

			void forward_and_grad(hidden_layer::linear_layer& layer);
		};

		struct softmax_optimizer :vector_optimizer
		{
		public:
			softmax_optimizer(size_t size);

			void forward_and_grad();
		};
	}
}

#endif