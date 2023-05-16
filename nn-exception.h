// This header defines exceptions in nn-cpp-lib

#ifndef NN_EXCEPTION_H
#define NN_EXCEPTION_H

#include <exception>
#include <string>
#include <format>

namespace nn
{
	// logic errors; eg. wrong file format
	class logic_exception :public std::exception
	{
	private:
		std::string message;
		std::string func_name;
		unsigned int func_line;

		std::string formatted_what;

		// format string for what()
		inline void format_what() 
		{
			formatted_what = std::format("network_logic_exception: {} (func:{}(), line:{})", message, func_name, func_line);
		}

	public:
		logic_exception(std::string msg, std::string func_name, unsigned int func_line) : message(msg), func_name(func_name), func_line(func_line)
		{
			format_what();
		}

		~logic_exception() throw () {}

		inline virtual const char* what() const throw()
		{
			return formatted_what.c_str();
		}
	};

	// numeric errors; eg. over-flow, under-flow, nan, out-of-range
	class numeric_exception :public std::exception
	{
	private:
		std::string message;
		std::string func_name;
		unsigned int func_line;

		std::string formatted_what;

		// format string for what()
		inline void format_what()
		{
			formatted_what = std::format("network_numeric_exception: {} (func:{}, line:{})", message, func_name, func_line);
		}

	public:
		numeric_exception(std::string msg, std::string func_name, unsigned int func_line) : message(msg), func_name(func_name), func_line(func_line)
		{
			format_what();
		}

		~numeric_exception() throw () {}

		inline virtual const char* what() const throw()
		{
			return formatted_what.c_str();
		}
	};

	// exception for memory allocation
	class memory_exception :public std::exception
	{
	private:
		std::string func_name;
		unsigned int func_line;

		std::string formatted_what;

		size_t requested_size;

		// format string for what()
		inline void format_what()
		{
			formatted_what = std::format("network_memory_exception: failed to allocate size {}bytes (func:{}, line:{})", requested_size, func_name, func_line);
		}

	public:
		memory_exception(size_t requested_size, std::string func_name, unsigned int func_line) : func_name(func_name), func_line(func_line), requested_size(requested_size)
		{
			format_what();
		}

		~memory_exception() throw () {}

		inline virtual const char* what() const throw()
		{
			return formatted_what.c_str();
		}
	};
}

#endif