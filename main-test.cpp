#include "nn-cpp-lib.h"

#include <iostream>
#include <math.h>
#include <format>

//== helper functions

void print_vector(const nn::vector& v)
{
	std::cout << v.format_str() << std::endl;
}

template<typename T>
void get_input_number(std::string prefix, T& num)
{
	std::cout << prefix << "=";
	std::cin >> num;
}

std::string get_line(std::string prefix)
{
	std::string str;
	std::cout << prefix << "=";
	std::getline(std::cin, str);
	return str;
}

int main()
{
	

	return 0;
}