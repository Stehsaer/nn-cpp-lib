// FILENAME: nn-layer-h
// Defines different layers and optimizers for a common neural-network
// Defines a base class, useful for creating a network

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
		struct conv2_linear_adapter_layer;
		struct conv2_layer;
		struct conv3_layer;
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

			void push_input(const vector & data);

			vector& get_input();
			size_t get_size();
		};

		struct matrix_input
		{
		private:
			matrix input;
			size_t input_width, input_height;

		public:
			matrix_input(size_t w, size_t h);

			void push_input(const matrix& data);

			matrix& get_input();
			size_t get_height();
			size_t get_width();
		};

		struct tensor_input
		{
		private:
			tensor input;
			size_t input_channel, input_width, input_height;

		public:
			tensor_input(size_t w, size_t h, size_t c);

			void push_input(const tensor& data);

			tensor& get_input();
			size_t get_channel();
			size_t get_height();
			size_t get_width();
		};
	}

	namespace hidden_layer
	{
		struct linear_layer
		{
		private:
			vector value, gradient;
			std::vector<vector> weights;
			size_t num_weights, num_neurons;
			float bias;

		public:
			linear_layer(size_t size, size_t weight_size); // size: number of neurons; weight_size: number of weights of each neuron

			void forward(input_layer::vector_input* prev, const activate_func* func);
			void forward(linear_layer* prev, const activate_func* func);

			void backward(optimizer::vector_optimizer* optimizer);
			void backward(linear_layer* last);

			void update_weights(input_layer::vector_input* prev, const activate_func* func, float learning_rate);
			void update_weights(hidden_layer::linear_layer* prev, const activate_func* func, float learning_rate);

			vector& get_value();
			vector& get_gradient();
			vector& get_weight(size_t index); // weight vector for neuron[index]
			size_t get_size();

			void rand_weights(float min, float max);
		};

		struct conv2_layer
		{
		private:
			matrix* kernals; // convolution kernals
			matrix* maps, * gradients; // maps: convolution result; gradients: as the word says
			size_t stride, padding;
			float* bias;

		public:
			const size_t w, h, depth, kernal_size; // in conv-related layers we choose to expose these parameters as const

			conv2_layer(size_t w, size_t h,size_t depth, size_t kernal_size, size_t stride = 1, size_t padding = 0);
			~conv2_layer(); // manual release memory

			void forward(input_layer::matrix_input* prev);
			void forward(hidden_layer::conv2_layer* prev);
			void forward(hidden_layer::relu_layer* prev);

			void backward(hidden_layer::conv2_layer* last);
			void backward(hidden_layer::relu_layer* last);

			matrix& get_kernal(size_t idx);
			matrix& get_map(size_t idx);
			matrix& get_gradient(size_t idx);

			void rand_weights(float min, float max);
		};

		struct relu_layer
		{
		private:
			matrix* maps, * gradients;
			
		public:
			const size_t w, h, depth;

			relu_layer(size_t w, size_t h, size_t depth);
			~relu_layer();

			void forward(hidden_layer::conv2_layer* prev);

			void backward(hidden_layer::conv2_layer* prev);

			matrix& get_map(size_t idx);
			matrix& get_gradient(size_t idx);
		};

		struct conv3_layer
		{
		private:
			tensor* kernals;
			matrix* feature_maps, feature_gradient;
			size_t stride, padding;
			size_t w, h, channel, depth, kernal_size;

		public:
			conv3_layer(size_t w, size_t h, size_t channel, size_t depth, size_t kernal_size, size_t stride = 1, size_t padding = 0);
		};
	}

	namespace optimizer
	{
		struct vector_optimizer
		{
		protected:
			vector output, gradient, target;
			size_t optimizer_size = 0;
			float loss;

		public:
			vector& get_gradient();
			vector& get_output();
			size_t get_size();
			float get_loss();
			void push_target(const vector & target);

			virtual void forward_and_grad(hidden_layer::linear_layer* layer) = 0;
			virtual void forward(hidden_layer::linear_layer* layer) = 0;
		};

		struct mse_optimizer :vector_optimizer
		{
		public:
			mse_optimizer(size_t size);

			void forward_and_grad(hidden_layer::linear_layer* layer);
			void forward(hidden_layer::linear_layer* layer);
		};

		struct softmax_optimizer :vector_optimizer
		{
		public:
			softmax_optimizer(size_t size);

			void forward_and_grad(hidden_layer::linear_layer* layer);
			void forward(hidden_layer::linear_layer* layer);
		};
	}

	template<typename input_T, typename output_T>
	class base_network
	{
	public:
		float learning_rate = 0.01f;

		virtual void feed_data(const input_T& input) = 0;
		virtual output_T& get_output() = 0;

		virtual void forward_and_grad(const output_T& target) = 0;
		virtual void backward() = 0;
		virtual void forward() = 0;
		virtual void update_weights() = 0;

		virtual float get_loss() = 0;

		virtual void init_weights(float min, float max) = 0;
	};
}

#endif