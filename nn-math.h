// This header defines common mathmatic operation functions

#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn-exception.h"

#include <vector>

namespace nn
{
	namespace math
	{
		float dot(float* left, float* right, size_t num); // inner-product of two vectors (intended for avx acceleration)
	}

	template<typename T>
	struct search_result
	{
	public:
		T result;
		size_t index;

		search_result(T result, size_t index): result(result), index(index){}
	};

	// a float type vector, 1 dimensional
	struct vector
	{
	private:
		size_t vector_size;
		float* vector_data;

	public:
		vector(size_t size); // initialize using size
		vector(std::initializer_list<float> initializer); // initialize using initializer_list
		vector(const vector& src); // copy construction function
		vector(vector&& src) noexcept; // move construction function
		~vector();

		void free(); // free data

		constexpr const size_t size(); // dimension of the vector
		constexpr float& operator [](size_t idx); // number at the index
		constexpr float* data(); // never use this unless necessary

		void fill(float num);

		static float dot(const vector& left, const vector& right); // dot product
		void operator /=(float num); // calculate average
		void operator +=(vector& v); // add per element

		std::string format_str(); // get formatted string as format: "vector(0.0,0.0...)"

		search_result<float> max(); // get largest element
		search_result<float> min(); // get largest element
	};

	// 2d matrix, float format
	struct matrix
	{
	private:
		size_t w, h;
		float* matrix_data;
	
	public:
		matrix(size_t w, size_t h);
		matrix(const matrix& src);
		matrix(matrix&& src) noexcept;
		~matrix();

		void free(); // free data

		constexpr const size_t width(); // matrix width
		constexpr const size_t height(); // matrix height
		constexpr float& at(size_t x, size_t y); // number at position(x,y)
		constexpr float* data(); // never use this unless necessary

		void fill(float num);

		vector to_vector();
	};
}

#endif