// This header defines common mathmatic operation functions

#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn-exception.h"

#include <vector>
#include <functional>

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
		vector();
		vector(size_t size); // initialize using size
		vector(std::initializer_list<float> initializer); // initialize using initializer_list
		vector(const vector& src); // copy construction function
		vector(vector&& src) noexcept; // move construction function
		~vector();

		const size_t size(); // dimension of the vector
		float& operator [](size_t idx); // number at the index
		float* data(); // never use this unless necessary

		void fill(float num);

		static float dot(const vector& left, const vector& right); // dot product
		void operator /=(float num); // calculate average
		void operator +=(vector& v); // add per element

		vector& operator =(const vector& src);

		std::string format_str(); // get formatted string as format: "vector(0.0,0.0...)"

		search_result<float> max(); // get largest element
		search_result<float> min(); // get largest element
		
		void do_foreach(std::function<void(size_t)> func); // execute operation foreach element
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
		matrix(const matrix& src);
		matrix(matrix&& src) noexcept;
		~matrix();

		const size_t width(); // matrix width
		const size_t height(); // matrix height
		float& at(size_t x, size_t y); // number at position(x,y)
		float* data(); // never use this unless necessary

		matrix& operator =(const matrix& src);

		void fill(float num);

		vector to_vector();

		void do_foreach(std::function<void(size_t, size_t)> func); // execute operation foreach element (x,y)
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

		const size_t channels(); // tensor channels
		const size_t height(); // tensor height
		const size_t width(); // tensor width
		float& at(size_t x, size_t y, size_t channel); // number at (channel,x,y)
		matrix& channel(size_t channel); // returns the matrix at given channel

		tensor& operator =(const tensor& src);

		void fill(float num);
	};

	namespace math
	{
		//== Basic algorithms

		float dot(float* left, float* right, size_t num); // inner-product of two vectors (intended for avx acceleration)

		//== Helper functions

		vector one_hot(size_t max_one_hot, size_t label); // one-hot helper for vector
		void rand_vector(vector& v, float min, float max);
	}
}

#endif