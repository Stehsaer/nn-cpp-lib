#include "nn-math.h"

#include <random>

float nn::math::dot(float* left, float* right, size_t num)
{
	float result = 0.0f;
	for (size_t i = 0; i < num; i++)
		result += left[i] * right[i];
	return result;
}

nn::vector nn::math::one_hot(size_t max_one_hot, size_t label)
{
	if (label >= max_one_hot || max_one_hot == 0)
		throw nn::numeric_exception("one-hot error", __FUNCTION__, __LINE__);

	vector v(max_one_hot);
	v.fill(0.0f);
	v[label] = 1.0f;
	return v;
}

void nn::math::rand_vector(vector& v, float min, float max)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution dist(min, max);

	for (size_t i = 0; i < v.size(); i++)
	{
		v[i] = dist(mt);
	}
}

nn::vector::vector()
{
	vector_size = 0;
	vector_data = nullptr;
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

size_t nn::vector::size() const
{
	return vector_size;
}

float& nn::vector::operator[](size_t idx) const
{
	_ASSERT(idx < vector_size);

	return vector_data[idx];
}

float* nn::vector::data()
{
	return vector_data;
}

void nn::vector::fill(float num)
{
	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] = num;
}

float nn::vector::dot(vector& left, vector& right)
{
	if (left.vector_size != right.vector_size)
		throw nn::numeric_exception("vector dimension mismatch", __FUNCTION__, __LINE__);

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
		throw nn::numeric_exception("vector dimension mismatch", __FUNCTION__, __LINE__);

	for (size_t i = 0; i < vector_size; i++)
		vector_data[i] += v[i];
}

nn::vector& nn::vector::operator=(const nn::vector& src)
{
	if (vector_size != src.vector_size)
	{
		delete[] vector_data;
		vector_size = src.vector_size;
		vector_data = new float[vector_size];
	}

	memcpy(vector_data, src.vector_data, sizeof(float) * vector_size);

	return *this;
}

std::string nn::vector::format_str() const
{
	std::string formatted = "vector(";

	for (size_t i = 0; i < vector_size; i++)
	{
		formatted += std::format("{:.5f}{}", vector_data[i], i == vector_size - 1 ? "" : ", ");
	}

	formatted += ")";

	return formatted;
}

nn::search_result<float> nn::vector::max() const
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

nn::search_result<float> nn::vector::min() const
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

float nn::vector::sum() const
{
	float sum = 0.0f;
	for (size_t i = 0; i < vector_size; i++)
	{
		sum += vector_data[i];
	}
	return sum;
}

void nn::vector::for_each(std::function<void(size_t, float&)> func)
{
	for (size_t i = 0; i < vector_size; i++)
		func(i, vector_data[i]);
}

nn::matrix::matrix()
{
	w = 0;
	h = 0;
	matrix_data = nullptr;
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

size_t nn::matrix::width() const
{
	return w;
}

size_t nn::matrix::height() const
{
	return h;
}

float& nn::matrix::at(size_t x, size_t y)
{
	_ASSERT(x < w && y < h);

	return matrix_data[y * w + x];
}

float nn::matrix::at(size_t x, size_t y) const
{
	_ASSERT(x < w && y < h);

	return matrix_data[y * w + x];
}

float* nn::matrix::data()
{
	return matrix_data;
}

nn::matrix& nn::matrix::operator=(const matrix& src)
{
	if (w != src.w || h != src.h)
	{
		delete[] matrix_data;

		w = src.w;
		h = src.h;

		matrix_data = new float[w * h];
	}

	memcpy(matrix_data, src.matrix_data, sizeof(float) * w * h);

	return *this;
}

void nn::matrix::fill(float num)
{
	for (size_t i = 0; i < w * h; i++)
	{
		matrix_data[i] = num;
	}
}

nn::vector nn::matrix::to_vector() const
{
	nn::vector v(w * h);
	memcpy(v.data(), matrix_data, sizeof(float) * w * h);

	return v;
}

void nn::matrix::for_each(std::function<void(size_t, size_t, float&)> func)
{
	for (size_t x = 0; x < w; x++)
		for (size_t y = 0; y < h; y++)
			func(x, y, matrix_data[y * w + x]);
}

nn::tensor::tensor()
{
	c = w = h = 0;
}

nn::tensor::tensor(size_t c, size_t w, size_t h) :c(c), w(w), h(h)
{
	for (size_t i = 0; i < c; i++)
	{
		matrices.push_back(matrix(w, h));
	}
}

nn::tensor::tensor(const tensor& src)
{
	c = src.c;
	w = src.w;
	h = src.h;

	matrices = src.matrices;
}

nn::tensor::tensor(tensor&& src) noexcept
{
	c = src.c;
	w = src.w;
	h = src.h;

	matrices.assign(src.matrices.begin(), src.matrices.end());
}

nn::tensor::~tensor()
{
	matrices.clear();
}

size_t nn::tensor::channels() const
{
	return c;
}

size_t nn::tensor::height() const
{
	return h;
}

size_t nn::tensor::width() const
{
	return w;
}

float& nn::tensor::at(size_t x, size_t y, size_t channel)
{
	return matrices[channel].at(x, y);
}

float nn::tensor::at(size_t x, size_t y, size_t channel) const
{
	return matrices[channel].at(x,y);
}

nn::matrix& nn::tensor::channel(size_t channel)
{
	return matrices[channel];
}

nn::tensor& nn::tensor::operator=(const tensor& src)
{
	c = src.c;
	w = src.w;
	h = src.h;

	for (size_t i = 0; i < c; i++)
	{
		if (i < matrices.size()) // channel exists, try to avoid re-allocating
			matrices[i] = src.matrices[i];
		else // channel doesn't exist, creat a new one
			matrices.push_back(matrix(src.matrices[i]));
	}

	matrices.shrink_to_fit(); // shrink to reduce data usage

	return *this;
}

void nn::tensor::fill(float num)
{
	for (auto& matrix : matrices)
		matrix.fill(num);
}