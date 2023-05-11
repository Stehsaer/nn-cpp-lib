#include "nn-math.h"

float nn::math::dot(float* left, float* right, size_t num)
{
	float result = 0.0f;
	for (size_t i = 0; i < num; i++)
		result += left[i] * right[i];
	return result;
}

nn::vector::vector(size_t size): vector_size(size)
{
	vector_data = new float[size];
}

nn::vector::vector(std::initializer_list<float> initializer)
{
	vector_size = initializer.size();
	vector_data = new float[vector_size];

	for (size_t i = 0; i < vector_size; i++)
	{
		vector_data[i] = *(initializer.begin() + i);
	}
}

nn::vector::vector(vector&& src) noexcept
{
	vector_size = src.vector_size;
	vector_data = src.vector_data;
}

void nn::vector::free()
{
	if (vector_data)
		delete[] vector_data;
	else
		throw nn::network_logic_exception("vector data invalid", __FUNCTION__, __LINE__);
}

constexpr const size_t nn::vector::size()
{
	return vector_size;
}

float& nn::vector::operator[](size_t idx)
{
	if (idx >= vector_size)
		throw nn::network_numeric_exception("vector out-of-range", __FUNCTION__, __LINE__);

	return vector_data[idx];
}

constexpr float* nn::vector::data()
{
	return vector_data;
}

float nn::vector::dot(const vector& left, const vector& right)
{
	if(left.vector_size != right.vector_size)
		throw nn::network_numeric_exception("vector dimension mismatch", __FUNCTION__, __LINE__);

	return nn::math::dot(left.vector_data, right.vector_data, left.vector_size);
}

void nn::vector::operator/=(float num)
{
	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] /= num;
}

void nn::vector::operator+=(vector& v)
{
	if(v.vector_size != vector_size )
		throw nn::network_numeric_exception("vector dimension mismatch", __FUNCTION__, __LINE__);

	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] += v[i];
}
