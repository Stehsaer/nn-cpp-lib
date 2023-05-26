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

nn::cifar10_dataset::~cifar10_dataset()
{
	for (auto ptr : set)
		delete ptr;

	set.clear();
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

nn::mnist_dataset::~mnist_dataset()
{
	for (auto ptr : set)
		delete ptr;

	set.clear();
}

void nn::mnist_dataset::flip_all()
{
	for (auto ptr : set)
	{
		math::flip_matrix_square(ptr->get_data());
	}
}

//== mnist helper functions, local

namespace mnist_helper
{
	constexpr int mnist_data_mgnum = 2051;
	constexpr int mnist_label_mgnum = 2049;

	constexpr size_t minimum_data_size = 16;
	constexpr size_t minimum_label_size = 8;

	// read int from mnist dataset
	int mnist_read_int(unsigned char* ptr)
	{
		int i = *((int*)ptr);
		if (nn::math::little_endian)
			return nn::math::reverse_order_int(i);
		else
			return i;
	}
}

void nn::mnist_dataset::add_source(std::string data_path, std::string label_path)
{
	auto data_file = file::file_read_bytes(data_path);
	auto label_file = file::file_read_bytes(label_path);

	if (!data_file.valid || !label_file.valid)
		throw logic_exception("can't read file", __FUNCTION__, __LINE__);

	// check file size, at least 8 bytes for label file, 16 bytes for image file

	if (data_file.size <= mnist_helper::minimum_data_size || label_file.size <= mnist_helper::minimum_label_size)
	{
		throw numeric_exception("input file obviously too small", __FUNCTION__, __LINE__);
	}

	auto data_ptr = data_file.data; // start ptr
	auto label_ptr = label_file.data;

	// check magic number
	if (mnist_helper::mnist_read_int(data_ptr) != mnist_helper::mnist_data_mgnum || mnist_helper::mnist_read_int(label_ptr) != mnist_helper::mnist_label_mgnum)
	{
		throw numeric_exception("input file magic number mismatch", __FUNCTION__, __LINE__);
	}

	// acquire basic parameters

	int mnist_image_count = mnist_helper::mnist_read_int(data_ptr + 4); // number of mnist images
	int mnist_w = mnist_helper::mnist_read_int(data_ptr + 8);
	int mnist_h = mnist_helper::mnist_read_int(data_ptr + 12);

	if (mnist_image_count != mnist_helper::mnist_read_int(label_ptr + 4)) // check if data and label have the same image count
	{
		throw numeric_exception("mismatched image count", __FUNCTION__, __LINE__);
	}
	if (mnist_image_count <= 0 || mnist_w <= 0 || mnist_h <= 0) // check range
	{
		throw numeric_exception("excepting positive integers", __FUNCTION__, __LINE__);
	}

	// check file size
	if (data_file.size != 16 + static_cast<size_t>(mnist_w) * mnist_h * mnist_image_count 
		|| label_file.size != 8 + static_cast<unsigned long long>(mnist_image_count))
	{
		throw numeric_exception("mismatched file size", __FUNCTION__, __LINE__);
	}

	size_t largest_label = 0;

	for (size_t img_index = 0; img_index < mnist_image_count; img_index++)
	{
		// get label
		size_t label = *(label_ptr + 8 + img_index);

		// find largest label
		if (label > largest_label)
			largest_label = label;

		// get img pixels
		nn::matrix img(mnist_w, mnist_h);
		img.for_each([mnist_w, mnist_h, img_index, data_ptr](size_t x, size_t y, float& num)
			{
				num = *(data_ptr + 16 + static_cast<size_t>(mnist_w) * mnist_h * img_index + y * mnist_w + x) / 255.0f;// C1001??
			});

		set.push_back(new nn::mnist_data(img, label));
	}

	for (auto ptr : set)
		ptr->gen_targets(largest_label);
}

nn::mnist_data::mnist_data(const nn::matrix& m, size_t label)
{
	data = m;
	data_label = label;
}

size_t nn::mnist_data::get_label()
{
	return data_label;
}

void nn::mnist_data::gen_targets(size_t max_label)
{
	target = math::one_hot(max_label, data_label);
}
