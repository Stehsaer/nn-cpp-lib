// This header defines common mathmatic operation functions

#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn-exception.h"

#include <vector>
#include <functional>
#include <bit>

namespace nn
{
	template<typename T>
	struct search_result
	{
	public:
		T result;
		size_t index;

		search_result(T result, size_t index) : result(result), index(index) {}
	};

	// a float type vector, 1 dimensional
	struct vector
	{
	private:
		size_t vector_size;
		float* vector_data;

	public:
		vector(); // initialize a placeholder with a size of 0 (invalid vector, don't use unless necessary)
		vector(size_t size); // initialize using size. Don't forget to initialize values manually or using fill()
		vector(std::initializer_list<float> initializer); // initialize using initializer_list
		vector(const vector& src); // copy construction function
		vector(vector&& src) noexcept; // move construction function
		~vector();

		size_t size() const; // dimension of the vector
		float& operator [](size_t idx) const; // number at the index
		float* data(); // never use this unless necessary

		void fill(float num);

		static float dot(vector& left, vector& right); // dot product
		void operator /=(float num); // calculate average
		void operator +=(vector& v); // add per element

		vector& operator =(const vector& src);
		vector& operator =(vector&& src) noexcept;
		vector& operator =(std::initializer_list<float> list);

		std::string format_str() const; // get formatted string as format: "vector(0.0,0.0...)"

		search_result<float> max() const; // get largest element
		search_result<float> min() const; // get largest element
		float sum() const;
		
		void for_each(std::function<void(size_t, float&)> func); // execute operation foreach element
	};

	// 2d matrix, float format
	struct matrix
	{
	private:
		size_t w, h;
		float* matrix_data;

	public:
		matrix();
		matrix(size_t w, size_t h);
		matrix(size_t w, size_t h, std::initializer_list<float> val);
		matrix(const matrix& src);
		matrix(matrix&& src) noexcept;
		~matrix();

		size_t width() const; // matrix width
		size_t height() const; // matrix height
		float& at(size_t x, size_t y); // number at position(x,y)
		float at(size_t x, size_t y) const;
		float* data(); // never use this unless necessary

		bool valid();

		matrix& operator =(const matrix& src);
		matrix& operator =(matrix&& src) noexcept;

		void fill(float num);

		vector to_vector() const;
		void export_image(std::string path, int quality = 90);
		static matrix matrix_from_image(std::string path);

		void for_each(std::function<void(size_t, size_t, float&)> func); // execute operation foreach element (x,y)
	};

	// 3d tensor structure, consisting of numerous matrices
	struct tensor
	{
	private:
		std::vector<matrix> matrices;
		size_t c, w, h;

	public:
		tensor();
		tensor(size_t c, size_t w, size_t h);
		tensor(const tensor& src);
		tensor(tensor&& src) noexcept;
		~tensor();

		size_t channels() const; // tensor channels
		size_t height() const; // tensor height
		size_t width() const; // tensor width
		float& at(size_t x, size_t y, size_t channel); // number at (channel,x,y)
		float at(size_t x, size_t y, size_t channel) const;
		matrix& channel(size_t channel); // returns the matrix at given channel

		tensor& operator =(const tensor& src);
		tensor& operator =(tensor&& src) noexcept;

		void export_image(std::string path, int quality = 90);
		static tensor tensor_from_image(std::string path);

		void fill(float num);
	};

	namespace math
	{
		//== Basic algorithms

		float dot(float* left, float* right, size_t num); // inner-product of two vectors (intended for avx acceleration)

		//== Helper functions

		vector one_hot(size_t max_one_hot, size_t label); // one-hot helper for vector
		void rand_vector(vector& v, float min, float max);
		void rand_matrix(matrix& m, float min, float max);
		float rand_float(float min, float max);

		//== Image operations

		void flip_matrix_square(matrix& m);
		matrix flip_matrix_any(const matrix& m);

		//== Convolution

		nn::matrix conv_2d(const nn::matrix& src, const nn::matrix& kernal, size_t stride = 1, size_t padding = 0);
		void conv_2d(nn::matrix& dst, const nn::matrix& src, const nn::matrix& kernal, size_t stride = 1, size_t padding = 0);
		nn::matrix conv_3d(const nn::tensor& src, const nn::tensor& kernal, size_t stride = 1, size_t padding = 0);

		//== Bit-level operations

		int reverse_order_int(int x); // reverse byte order in int

		const bool little_endian = std::endian::native == std::endian::little;
	}
}

#endif