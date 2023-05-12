#include "nn-math.h"

float nn::math::dot(float* left, float* right, size_t num)
{
	float result = 0.0f;
	for (size_t i = 0; i < num; i++)
		result += left[i] * right[i];
	return result;
}

nn::vector::vector(size_t size) : vector_size(size)
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

nn::vector::vector(const vector& src)
{
	vector_size = src.vector_size;
	vector_data = new float[vector_size];

	//copy data from source
	memcpy(vector_data, src.vector_data, sizeof(float) * vector_size);
}

nn::vector::vector(vector&& src) noexcept
{
	vector_size = src.vector_size;
	vector_data = src.vector_data;
	src.vector_data = nullptr; // set data to nullptr, releasing the memory
}

nn::vector::~vector()
{
	if (vector_data)
		delete[] vector_data;
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

constexpr float& nn::vector::operator[](size_t idx)
{
	_ASSERT(idx < vector_size);

	return vector_data[idx];
}

constexpr float* nn::vector::data()
{
	return vector_data;
}

void nn::vector::fill(float num)
{
	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] = num;
}

float nn::vector::dot(const vector& left, const vector& right)
{
	if (left.vector_size != right.vector_size)
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
	if (v.vector_size != vector_size)
		throw nn::network_numeric_exception("vector dimension mismatch", __FUNCTION__, __LINE__);

	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] += v[i];
}

std::string nn::vector::format_str()
{
	std::string formatted = "vector(";

	for (size_t i = 0; i < vector_size; i++)
	{
		formatted += std::format("{:.5f}{}", vector_data[i], i == vector_size - 1 ? "" : ", ");
	}

	formatted += ")";

	return formatted;
}

nn::search_result<float> nn::vector::max()
{
	nn::search_result<float> max(0, 0);

	for (size_t i = 0; i < vector_size; i++)
	{
		if (vector_data[i] > max.result)
		{
			max.index = i;
			max.result = vector_data[i];
		}
	}

	return max;
}

nn::search_result<float> nn::vector::min()
{
	nn::search_result<float> min(0, 0);

	for (size_t i = 0; i < vector_size; i++)
	{
		if (vector_data[i] < min.result)
		{
			min.index = i;
			min.result = vector_data[i];
		}
	}

	return min;
}

nn::matrix::matrix(size_t w, size_t h) :w(w), h(h)
{
	matrix_data = new float[w * h];
}

nn::matrix::matrix(const matrix& src)
{
	w = src.w;
	h = src.h;

	matrix_data = new float[w * h];
	
	memcpy(matrix_data, src.matrix_data, sizeof(float) * w * h);
}

nn::matrix::matrix(matrix&& src) noexcept
{
	w = src.w;
	h = src.h;
	matrix_data = src.matrix_data;

	src.matrix_data = nullptr;
}

nn::matrix::~matrix()
{
	if (matrix_data)
		delete[] matrix_data;
}

void nn::matrix::free()
{
	if (matrix_data)
		delete[] matrix_data;
}

constexpr const size_t nn::matrix::width()
{
	return w;
}

constexpr const size_t nn::matrix::height()
{
	return h;
}

constexpr float& nn::matrix::at(size_t x, size_t y)
{
	_ASSERT(x < w && y < h);

	return matrix_data[y * h + x];
}

constexpr float* nn::matrix::data()
{
	return matrix_data;
}

void nn::matrix::fill(float num)
{
	for (size_t i = 0; i < w * h; i++)
	{
		matrix_data[i] = num;
	}
}

nn::vector nn::matrix::to_vector()
{
	nn::vector v(w * h);
	memcpy(v.data(), matrix_data, sizeof(float) * w * h);

	return v;
}
