#define _CRT_SECURE_NO_WARNINGS

#include <array>

#include "nn-dataset.h"
#include "nn-file.h"

nn::cifar10_data::cifar10_data(const nn::tensor& t, size_t label)
{
	data = t;
	data_label = label;
	target = nn::math::one_hot(10, label);
}

size_t nn::cifar10_data::get_label()
{
	return data_label;
}

void nn::cifar10_dataset::add_source(const std::string file_path)
{
	nn::file::file_binary_data binary_data = nn::file::file_read_bytes(file_path);

	if (!binary_data.valid)
		return;

	if (binary_data.size != 30730000)
		return;

	set.reserve(set.size() + 10000);

	for (size_t i = 0; i < 10000; i++)
	{
		unsigned char const* ptr = binary_data.data + i * 3073;

		// store label
		size_t label = *ptr;
		ptr++;

		// get data
		tensor data(3, 32, 32); // tensor, 3(channels)*32(width)*32(height)

		for (size_t c = 0; c < 3; c++)
		{
			auto& mat = data.channel(c);

			for (size_t x = 0; x < 32; x++)
				for (size_t y = 0; y < 32; y++)
					mat.at(x, y) = ptr[x * 32 + y] / 255.0f;

			ptr += 1024;
		}

		set.push_back(new nn::cifar10_data(data, label));
	}
}

std::string nn::cifar10_dataset::get_label(size_t index)
{
	static const std::array<std::string, 10> labels = { "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" };
	return labels[index];
}

void nn::cifar10_dataset::export_image(std::string path, size_t index, int quality)
{
	if (index >= set.size())
	{
		throw nn::numeric_exception("index out-of-range", __FUNCTION__, __LINE__);
	}

	set[index]->get_data().export_image(path, quality);
}
