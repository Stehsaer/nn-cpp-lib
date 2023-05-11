// This header defines exceptions in nn-cpp-lib

#ifndef NN_EXCEPTION_H
#define NN_EXCEPTION_H

#include <exception>
#include <string>
#include <format>

namespace nn
{
	// logic errors; eg. wrong file format
	class network_logic_exception :public std::exception
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
		network_logic_exception(std::string msg, std::string func_name, unsigned int func_line) : message(msg), func_name(func_name), func_line(func_line)
		{
			format_what();
		}

		~network_logic_exception() throw () {}

		inline virtual const char* what() const throw()
		{
			return formatted_what.c_str();
		}
	};

	// numeric errors; eg. over-flow, under-flow, nan, out-of-range
	class network_numeric_exception :public std::exception
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
		network_numeric_exception(std::string msg, std::string func_name, unsigned int func_line) : message(msg), func_name(func_name), func_line(func_line)
		{
			format_what();
		}

		~network_numeric_exception() throw () {}

		inline virtual const char* what() const throw()
		{
			return formatted_what.c_str();
		}
	};
}

#endif