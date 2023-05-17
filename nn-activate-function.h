#ifndef NN_ACTIVATE_FUNCTION_H
#define NN_ACTIVATE_FUNCTION_H

namespace nn
{
	class activate_func
	{
	public:
		virtual float forward(float x) const = 0;
		virtual float backward(float x) const = 0;
	};

	class relu_func :public activate_func
	{
	public:
		relu_func() {};
		float forward(float x) const;
		float backward(float x) const;
	};

	class leaky_relu_func :public activate_func
	{
	public:
		leaky_relu_func(){}
		float forward(float x) const;
		float backward(float x) const;
	};

	class sigmoid_func :public activate_func
	{
	public:
		sigmoid_func(){}
		float forward(float x) const;
		float backward(float x) const;
	};
}

#endif