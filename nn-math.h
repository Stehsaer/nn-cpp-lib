// This header defines common mathmatic operation functions

#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn-exception.h"

namespace nn
{
	namespace math
	{
		float dot(float* left, float* right, size_t num); // inner-product of two vectors (intended for avx acceleration)
	}

	// a float type vector
	struct vector
	{
	private:
		size_t vector_size;
		float* vector_data;

	public:
		vector(size_t size); // initialize using size
		vector(std::initializer_list<float> initializer); // initialize using initializer_list
		vector(const vector& src) = delete;
		vector(vector&& src) noexcept; // move construction function

		void free();

		constexpr const size_t size(); // dimension of the vector
		float& operator [](size_t idx); // number at the index
		constexpr float* data(); // never use this unless necessary or high-performance required

		static float dot(const vector& left, const vector& right); // dot product
		void operator /=(float num); // calculate average
		void operator +=(vector& v); // add per element
	};
}

#endif