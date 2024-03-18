// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_SPLIT_DENSE_H_
#define ANMLRIDDLE_SPLIT_DENSE_H_

#include <anmlriddle/com.h>

#include <sodium.h>

namespace anmlriddle {

// template <typename DerivedA, typename DerivedB>
// void SplitDense(const Eigen::Ref<const Eigen::MatrixX<Com>>& secret,
//                 Eigen::MatrixBase<DerivedA>& first_share, Eigen::MatrixBase<DerivedB>& second_share);
template <typename DerivedA, typename DerivedB>
inline void SplitDense(const Eigen::Ref<const Eigen::MatrixX<Com>>& secret,
                Eigen::MatrixBase<DerivedA>& first_share, Eigen::MatrixBase<DerivedB>& second_share) {
  if (secret.size() != first_share.size()
      || first_share.size() != second_share.size()
      || second_share.size() != secret.size()) {
    throw std::invalid_argument(
        "SplitVector: the secret and the two shares are of different sizes");
  }

  // Fill one share with random bytes
  randombytes_buf(first_share.derived().data(), first_share.size() * sizeof(Com));
  
  // Derive the other share from it
  second_share = secret - first_share;
}

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_SPLIT_DENSE_H_
