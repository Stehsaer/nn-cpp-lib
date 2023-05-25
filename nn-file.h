// FILENAME: nn-file.h
// Provide handful helper functions for reading and writing text and binary files
// Reading a file returns a struct file_binary_data

#ifndef NN_FILE_H
#define NN_FILE_H

#include <string>
#include <filesystem>

namespace nn::file
{
	struct file_binary_data
	{
	public:
		bool valid;
		unsigned char* data;
		size_t size;

		file_binary_data(unsigned char* data, size_t size) :data(data), size(size), valid(true) {}
		file_binary_data() :data(nullptr), size(0), valid(false) {}
		~file_binary_data();
	};

	bool file_exists(std::string path);
	size_t file_size(std::string path);

	file_binary_data file_read_bytes(std::string path);
	std::string file_read_string(std::string path);

	bool file_write_bytes(std::string path, file_binary_data data); // return true if success
	bool file_write_string(std::string path, std::string content); // return true if success
}

#endif