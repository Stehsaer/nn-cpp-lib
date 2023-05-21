#include "nn-cpp-lib.h"

#include <iostream>
#include <math.h>
#include <format>

void print_vector(const nn::vector& v)
{
	std::cout << v.format_str() << std::endl;
}

int main()
{
	std::string path;
	std::cin >> path;
	nn::matrix m = nn::matrix::matrix_from_image(path);
	nn::matrix kernal(3, 3, { 0,0.125,0,0.125,0.5,0.125,0,0.125,0 });
	nn::matrix result = nn::math::conv_2d(m, kernal, 2, 5);
	result.export_image("save.jpg");
	system("./save.png");

	return 0;
}