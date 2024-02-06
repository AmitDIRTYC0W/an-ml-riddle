// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_SPLIT_VECTOR_H_
#define ANMLRIDDLE_SPLIT_VECTOR_H_

#include <utility>
#include <ranges>

#include <anmlriddle/com.h>

namespace amr {

//SplitVector(const flatbuffers::Vector<short int>*, std::pair<std::vector<short int>, flatbuffers::Offset<flatbuffers::Vector<short int> > >)'
/*
template <ComRange Input, typename ContiguousOutput, typename Output>
requires std::ranges::input_range<Input>
//         && std::ranges::contiguous_range<ContiguousOutput>
//         && std::ranges::output_range<ContiguousOutput, Com>
//         && std::ranges::output_range<Output, Com>
// TODO we need those above comments!
void SplitVector(const Input* secret, std::pair<ContiguousOutput, Output> shares);
*/

/*
template <typename Input, typename FirstShare, typename SecondShare>
void SplitDense(const Input& secret,
                std::pair<Eigen::Ref<FirstShare>, Eigen::Ref<SecondShare>> shares);*/

template <typename T>
void SplitDense(const Eigen::Ref<const T>& secret, std::pair<Eigen::Ref<T>, Eigen::Ref<T>> shares);

}  // namespace amr

#endif  // ANMLRIDDLE_SPLIT_VECTOR_H_
