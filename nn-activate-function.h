#ifndef NN_ACTIVATE_FUNCTION_H
#define NN_ACTIVATE_FUNCTION_H

namespace nn
{
	class activate_func
	{
	public:
		virtual float forward(float x) = 0;
		virtual float backward(float x) = 0;
	};

	class relu_func :activate_func
	{
	public:
		float forward(float x);
		float backward(float x);
	};

	class leaky_relu_func :activate_func
	{
	public:
		float forward(float x);
		float backward(float x);
	};

	class sigmoid_func :activate_func
	{
	public:
		float forward(float x);
		float backward(float x);
	};
}

#endif