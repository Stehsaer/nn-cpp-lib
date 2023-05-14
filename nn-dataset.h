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
		~nn_data();

		DAT& get_data();
		TGT& get_target();
	};

	struct cifar10_data : public nn_data<nn::tensor, nn::vector>
	{
	public:
		cifar10_data(const nn::tensor& t, size_t label);
	};

	class cifar10_dataset
	{
	public:
		std::vector<cifar10_data*> set;

		static const std::string label_name[10];

		cifar10_dataset(const cifar10_dataset& src) = delete;
		cifar10_dataset(cifar10_dataset&& src) = delete;

		void add_source(std::string file_path);
		std::string get_label(size_t index);
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
	inline nn_data<DAT, TGT>::~nn_data()
	{
		delete &data;
		delete &target;
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