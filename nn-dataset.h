#ifndef NN_DATASET_H
#define NN_DATASET_H

#include <vector>
#include <string>

#include "nn-math.h"

namespace nn
{
	// base class for all datasets
	template<typename DAT, typename TGT>
	struct nn_data
	{
	protected:
		DAT data;
		TGT target;

	public:
		nn_data();
		nn_data(const DAT& dat, const TGT& tgt);
		nn_data(const nn_data& src) = delete; // NOTE: No copy or move of nn_data
		nn_data(nn_data&& src) = delete;

		DAT& get_data();
		TGT& get_target();
	};

	struct cifar10_data : public nn_data<nn::tensor, nn::vector>
	{
	private:
		size_t data_label;

	public:
		cifar10_data(const nn::tensor& t, size_t label);
		size_t get_label();

		void export_image(std::string path, int quality = 100);
	};

	class cifar10_dataset
	{
	public:
		std::vector<cifar10_data*> set;

		cifar10_dataset() {};
		cifar10_dataset(const cifar10_dataset& src) = delete;
		cifar10_dataset(cifar10_dataset&& src) = delete;
		~cifar10_dataset();

		void add_source(const std::string file_path);
		static std::string get_label(size_t index);

		void export_image(std::string path, size_t index, int quality = 100);
	};

	struct mnist_data :public nn_data<nn::matrix, nn::vector>
	{
	private:
		size_t data_label;

	public:
		mnist_data(const nn::matrix& m, size_t label);
		size_t get_label();

		void gen_targets(size_t max_label);

		void export_image(std::string path, int quality = 100);
	};

	class mnist_dataset
	{
	public:
		std::vector<mnist_data*> set;

		mnist_dataset() {};
		mnist_dataset(const mnist_dataset& src) = delete;
		mnist_dataset(mnist_dataset&& src) = delete;
		~mnist_dataset();

		void flip_all(); // flip mnist data
		void add_source(std::string data_path, std::string label_path);
	};

	//== Inline function for template struct: nn_data

	template<typename DAT, typename TGT>
	inline nn_data<DAT, TGT>::nn_data()
	{
		data = DAT();
		target = TGT();
	}

	template<typename DAT, typename TGT>
	inline nn_data<DAT, TGT>::nn_data(const DAT& dat, const TGT& tgt)
	{
		data = dat;
		target = tgt;
	}

	template<typename DAT, typename TGT>
	inline DAT& nn_data<DAT, TGT>::get_data()
	{
		return data;
	}

	template<typename DAT, typename TGT>
	inline TGT& nn_data<DAT, TGT>::get_target()
	{
		return target;
	}
}

#endif