#include "nn-file.h"
#include "nn-exception.h"

#include <fstream>

namespace fs = std::filesystem;

bool nn::file::file_exists(std::string path)
{
    fs::path pth(path);

    if (fs::exists(pth))
    {
        return fs::is_regular_file(pth);
    }
    else
        return false;

}

size_t nn::file::file_size(std::string path)
{
    return fs::file_size(fs::path(path));
}

nn::file::file_binary_data nn::file::file_read_bytes(std::string path)
{
    std::ifstream stream = std::ifstream(path, std::ios::binary);

    if (stream.is_open())
    {
        size_t size = file_size(path);

        unsigned char* data;
        
        try
        {
             data = new unsigned char[size];
        }
        catch (std::bad_alloc&)
        {
            throw network_memory_exception(size, __FUNCTION__, __LINE__);
        }

        stream.read((char*)data, size);
        stream.close();

        return file_binary_data(data, size);
    }
    else
        return file_binary_data(); // return "invalid"
}

std::string nn::file::file_read_string(std::string path)
{
    std::ifstream stream = std::ifstream(path);

    if (stream.is_open())
    {
        std::string str;
        std::string temp;

        while (std::getline(stream, temp))
            str += temp + "\n";

        stream.close();

        return str;
    }
    else
        return "";
}

bool nn::file::file_write_bytes(std::string path, file_binary_data data)
{
    if (file_exists(path)) fs::remove(fs::path(path));

    std::ofstream stream = std::ofstream(path, std::ios::binary);

    if (stream.is_open())
    {
        stream.seekp(std::ios::beg);
        stream.write((const char*)data.data, data.size);
        stream.close();

        return true;
    }
    else
        return false;
}

bool nn::file::file_write_string(std::string path, std::string content)
{
    if (file_exists(path)) fs::remove(fs::path(path));
    
    std::ofstream stream = std::ofstream(fs::path(path));

    if (stream.is_open())
    {
        stream << content;
        stream.close();

        return true;
    }
    else
        return false;
}
