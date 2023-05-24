#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-img/stb_image_write.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb-img/stb_image.h"

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
	if (label >= max_one_hot + 1 || max_one_hot == 0)
		throw nn::numeric_exception("one-hot error", __FUNCTION__, __LINE__);

	vector v(max_one_hot + 1);
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

void nn::math::rand_matrix(matrix& m, float min, float max)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution dist(min, max);

	for (size_t i = 0; i < m.width() * m.height(); i++)
	{
		m.data()[i] = dist(mt);
	}
}

float nn::math::rand_float(float min, float max)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution dist(min, max);
	return dist(mt);
}

void nn::math::flip_matrix_square(matrix& m)
{
	if (m.width() != m.height())
		throw nn::numeric_exception("not a square matrix", __FUNCTION__, __LINE__);

	const nn::matrix temp = m;
	m.for_each([&temp](size_t x, size_t y, float& num)
		{
			num = temp.at(y, x);
		});
}

nn::matrix nn::math::flip_matrix_any(const matrix& m)
{
	nn::matrix out(m.height(), m.width());

	out.for_each([&m](size_t x, size_t y, float& num)
		{
			num = m.at(y, x);
		});

	return out;
}

nn::matrix nn::math::conv_2d(const nn::matrix& src, const nn::matrix& kernal, size_t stride, size_t padding)
{
	// check input parameters
	if (kernal.width() > src.width() || kernal.height() > src.height())
		throw nn::numeric_exception("(conv)kernal bigger than source", __FUNCTION__, __LINE__);
	if (stride == 0)
		throw nn::numeric_exception("(conv)stride should be larger than 0", __FUNCTION__, __LINE__);

	// calculate height and width
	size_t output_w = (src.width() - kernal.width() + padding * 2) / stride;
	size_t output_h = (src.height() - kernal.height() + padding * 2) / stride;

	nn::matrix out(output_w, output_h);

	conv_2d(out, src, kernal, stride, padding);

	return out;
}

void nn::math::conv_2d(nn::matrix& dst, const nn::matrix& src, const nn::matrix& kernal, size_t stride, size_t padding)
{
	// check input parameters
	if (kernal.width() > src.width() || kernal.height() > src.height())
		throw nn::numeric_exception("(conv)kernal bigger than source", __FUNCTION__, __LINE__);
	if (stride == 0)
		throw nn::numeric_exception("(conv)stride should be larger than 0", __FUNCTION__, __LINE__);

	// calculate height and width
	size_t output_w = (src.width() - kernal.width() + padding * 2) / stride;
	size_t output_h = (src.height() - kernal.height() + padding * 2) / stride;

	if (dst.width() != output_w || dst.height() != output_h)
		throw nn::numeric_exception("mismatch DST size", __FUNCTION__, __LINE__);

	// do conv
	dst.for_each([&src, &kernal, stride, padding](size_t x, size_t y, float& num)
		{
			num = 0.0f;

			for (size_t _x = 0; _x < kernal.width(); _x++) for (size_t _y = 0; _y < kernal.height(); _y++)
			{
				size_t src_x = _x + x * stride - padding;
				size_t src_y = _y + y * stride - padding;

				num += kernal.at(_x, _y) *
					(
						src_x < 0 || src_x >= src.width() || src_y < 0 || src_y >= src.height() // check padding area
						? 0.0f // padding area, fill with 0.0f
						: src.at(src_x, src_y)
						); // non-padding area, use real data
			}
		});
}

nn::matrix nn::math::conv_3d(const nn::tensor& src, const nn::tensor& kernal, size_t stride, size_t padding)
{
	// check input parameters
	if (src.channels() != kernal.channels())
		throw nn::numeric_exception("channel count mismatch", __FUNCTION__, __LINE__);
}

int nn::math::reverse_order_int(int x)
{
	return 
		(0xff000000 & x) >> 24
		| (0x00ff0000 & x) >> 8
		| (0x0000ff00 & x) << 8
		| (0x000000ff & x) << 24;
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

nn::vector& nn::vector::operator =(nn::vector&& src) noexcept
{
	delete[] vector_data;
	vector_size = src.vector_size;
	vector_data = src.vector_data;

	src.vector_data = nullptr;

	return *this;
}

nn::vector& nn::vector::operator =(std::initializer_list<float> list)
{
	if (list.size() != vector_size)
	{
		delete[] vector_data;
		vector_size = list.size();
		vector_data = new float[vector_size];
	}

	for (size_t i = 0; i < vector_size; i++)
	{
		vector_data[i] = *(list.begin() + i);
	}

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

nn::matrix::matrix(size_t w, size_t h, std::initializer_list<float> val) : w(w), h(h)
{
	if (val.size() != w * h)
		throw nn::numeric_exception("initializer value count mismatch!", __FUNCTION__, __LINE__);
	matrix_data = new float[w * h];

	for (size_t i = 0; i < w * h; i++)
	{
		matrix_data[i] = *(val.begin() + i);
	}
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

bool nn::matrix::valid()
{
	return matrix_data != nullptr && w > 0 && h > 0;
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

nn::matrix& nn::matrix::operator=(matrix&& src) noexcept
{
	delete[] matrix_data;
	
	w = src.w;
	h = src.h;
	
	matrix_data = src.matrix_data;
	src.matrix_data = nullptr;

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

void nn::matrix::export_image(std::string path, int quality)
{
	if (w > 65535 || h > 65535)
		throw nn::numeric_exception("width or height too large", __FUNCTION__, __LINE__);

	std::unique_ptr<unsigned char> dat(new unsigned char[w * h]);

	for (size_t i = 0; i < w * h; i++)
	{
		dat.get()[i] = unsigned char(matrix_data[i] * 255.0f);
	}

	stbi_write_jpg(path.c_str(), static_cast<int>(w), static_cast<int>(h), 1, dat.get(), quality);
}

nn::matrix nn::matrix::matrix_from_image(std::string path)
{
	int x, y, comp;
	unsigned char* data = stbi_load(path.c_str(), &x, &y, &comp, 1);
	if (!data)
		throw nn::logic_exception("read file failed", __FUNCTION__, __LINE__);

	matrix m(x, y);

	for (size_t i = 0; i < static_cast<size_t>(x) * y; i++)
	{
		m.data()[i] = data[i] / 255.0f;
	}

	free(data);

	return m;
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
	return matrices[channel].at(x, y);
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

nn::tensor& nn::tensor::operator=(tensor&& src) noexcept
{
	w = src.w;
	h = src.h;
	c = src.c;

	matrices.assign(src.matrices.begin(), src.matrices.end());

	return *this;
}

void nn::tensor::export_image(std::string path, int quality)
{
	if (c != 3)
		throw nn::logic_exception("tensor doesn't have exactly 3 channels", __FUNCTION__, __LINE__);

	std::unique_ptr<unsigned char> reordered_data(new unsigned char[3 * w * h]);

	// reorder data
	for (size_t c = 0; c < 3; c++)
		for (size_t x = 0; x < 32; x++)
			for (size_t y = 0; y < 32; y++)
				reordered_data.get()[x * h * 3 + y * 3 + c] = unsigned char(matrices[c].at(x, y) * 255.0f);

	if (!stbi_write_jpg(path.c_str(), 32, 32, 3, reordered_data.get(), quality))
		throw nn::logic_exception("write jpg failed", __FUNCTION__, __LINE__);
}

void nn::tensor::fill(float num)
{
	for (auto& matrix : matrices)
		matrix.fill(num);
}