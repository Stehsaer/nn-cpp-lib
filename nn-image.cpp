#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-img/stb_image_write.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb-img/stb_image.h"

#include "nn-image.h"

nn::matrix nn::image::image_load_matrix(std::string path)
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

nn::tensor nn::image::image_load_tensor(std::string path)
{
	int w, h, channel;
	unsigned char* img = stbi_load(path.c_str(), &w, &h, &channel, 3);

	nn::tensor out(w, h, 3);

	out.for_each([img, &out, w, h](size_t x, size_t y, size_t channel, float& num)
		{
			num = img[x * h * 3 + y * 3 + channel] / 255.0f;
		});

	return out;
}

void nn::image::image_save_tensor_jpg(const tensor& src, std::string path, int quality)
{
	if (src.channels() != 3)
		throw nn::logic_exception("tensor doesn't have exactly 3 channels", __FUNCTION__, __LINE__);

	std::unique_ptr<unsigned char> reordered_data(new unsigned char[3 * src.width() * src.height()]);

	// reorder data
	for (size_t c = 0; c < 3; c++)
		for (size_t x = 0; x < 32; x++)
			for (size_t y = 0; y < 32; y++)
				reordered_data.get()[x * src.height() * 3 + y * 3 + c] = unsigned char(src.at(x, y,c) * 255.0f);

	if (!stbi_write_jpg(path.c_str(), 32, 32, 3, reordered_data.get(), quality))
		throw nn::logic_exception("write jpg failed", __FUNCTION__, __LINE__);
}

void nn::image::image_save_matrix_jpg(const matrix& src, std::string path, int quality)
{
	if (src.width() > 65535 || src.height() > 65535)
		throw nn::numeric_exception("width or height too large", __FUNCTION__, __LINE__);

	std::unique_ptr<unsigned char> dat(new unsigned char[src.width() * src.height()]);

	for (size_t y = 0; y < src.height(); y++)
		for (size_t x = 0; x < src.width(); x++)
			dat.get()[y * src.width() + x] = unsigned char(src.at(x, y) * 255.0f);

	stbi_write_jpg(path.c_str(), static_cast<int>(src.width()), static_cast<int>(src.height()), 1, dat.get(), quality);
}
