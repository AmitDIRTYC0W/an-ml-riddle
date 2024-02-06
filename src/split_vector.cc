// Copyright 2023 Amit Goren

#include "split_vector.h"

#include <algorithm>
#include <execution>

#include <sodium.h>

namespace amr {

/*
template <ComRange Input, typename ContiguousOutput, typename Output>
requires std::ranges::input_range<Input>
         && std::ranges::contiguous_range<ContiguousOutput>
         && std::ranges::output_range<ContiguousOutput, Com>
         && std::ranges::output_range<Output, Com>
void SplitVector(Input secret, std::pair<ContiguousOutput, Output> shares) {
  if (secret.size() != shares.first.size()
      || shares.first.size() != shares.second.size()
      || secret.second.size() != secret.size()) {
    throw std::invalid_argument(
        "SplitVector: the secret and the two shares are of different sizes");
  }

  // Fill one share with random bytes
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));

  // other share = secret - our share
  std::transform(secret.cbegin(), secret.cend(), shares.first.cbegin(),
                 shares.second.begin(), std::minus<Com>());
}*/

// FIXME shouldn't be here...
/*
template <typename T>
concept ComObject = requires(T a) {
  { a(Eigen::Index) } -> std::convertible_to<Com>;
};*/

// FIXME define ComLikeDense, RawComDense, ComAssignableDense
// FIXME add requirements
//template <ComLikeDense Input, RawComDense FirstShare, ComAssignableDense SecondShare>

// TODO FirstShare must be Eigen::PlainObjectBase
/*
template <typename Input, typename FirstShare, typename SecondShare>
void SplitDense(const Input& secret,
                std::pair<Eigen::Ref<FirstShare>, Eigen::Ref<SecondShare>> shares) {
  if (secret.size() != shares.first.size()
      || shares.first.size() != shares.second.size()
      || secret.second.size() != secret.size()) {
    throw std::invalid_argument(
        "SplitVector: the secret and the two shares are of different sizes");
  }

  // Fill one share with random bytes
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));
  
  // Derive the other share from it
  shares.second = secret - shares.first;
}
*/
template <typename T>
void SplitDense(const Eigen::Ref<const T>& secret, std::pair<Eigen::Ref<T>, Eigen::Ref<T>> shares) {
  if (secret.size() != shares.first.size()
      || shares.first.size() != shares.second.size()
      || shares.second.size() != secret.size()) {
    throw std::invalid_argument(
        "SplitVector: the secret and the two shares are of different sizes");
  }

  // Fill one share with random bytes
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));
  
  // Derive the other share from it
  shares.second = secret - shares.first;
}

template void SplitDense<Eigen::VectorX<Com>>(const Eigen::Ref<const Eigen::VectorX<Com>>& secret, std::pair<Eigen::Ref<Eigen::VectorX<Com>>, Eigen::Ref<Eigen::VectorX<Com>>> shares);

template void SplitDense<Eigen::MatrixX<Com>>(const Eigen::Ref<const Eigen::MatrixX<Com>>& secret, std::pair<Eigen::Ref<Eigen::MatrixX<Com>>, Eigen::Ref<Eigen::MatrixX<Com>>> shares);

}  // namespace amr