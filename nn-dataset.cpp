#include "nn-dataset.h"
#include "nn-file.h"

nn::cifar10_data::cifar10_data(const nn::tensor& t, size_t label)
{
	data = t;
	target = nn::math::one_hot(10, label);
}

void nn::cifar10_dataset::add_source(std::string file_path)
{
	nn::file::file_binary_data binary_data = nn::file::file_read_bytes(file_path);

	if (!binary_data.valid)
		return;

	if (binary_data.size != 30730000)
		return;

	for (size_t i = 0; i < 10000; i++)
	{
		unsigned char label = binary_data.data[i * 3073];
	}
}

std::string nn::cifar10_dataset::get_label(size_t index)
{
	static const std::string labels[10] = { "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" };
	return labels[index];
}
