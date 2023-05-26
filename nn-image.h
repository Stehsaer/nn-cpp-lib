#ifndef NN_IMAGE_H
#define NN_IMAGE_H

#include <string>
#include "nn-math.h"

namespace nn::image
{
	nn::matrix image_load_matrix(std::string path);
	nn::tensor image_load_tensor(std::string path);

	void image_save_tensor_jpg(const tensor& src, std::string path, int quality = 95);
	void image_save_matrix_jpg(const matrix& src, std::string path, int quality = 95);

	nn::matrix image_upscale_matrix(const matrix& src, size_t scale_factor);
	nn::tensor image_upscale_tensor(const tensor& src, size_t scale_factor);
}

#endif