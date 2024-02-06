// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_
#define ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_

#include <Eigen/Dense>

#include <anmlriddle/com.h>
#include <anmlriddle/synchronised_queue.h>
#include "server_message_generated.h"
#include "client_message_generated.h"
#include "common_generated.h"

namespace amr {

// Multiplication using Beaver's triplets (Donald Beaver. Efficient
// Multiparty Protocols Using Circuit Randomization. CRYPTO 1991.)
//
// WARNING: Multiplication triplets shall not be re-used.
struct MultiplicationTriplet {
  Eigen::MatrixX<Com> a_share;
  Eigen::MatrixX<Com> b_share;
  Eigen::MatrixX<Com> c_share;
  
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  MultiplicationTriplet(const Eigen::MatrixBase<DerivedA>& a_share,
                        const Eigen::MatrixBase<DerivedB>& b_share,
                        const Eigen::MatrixBase<DerivedC>& c_share)
      : a_share(a_share), b_share(b_share), c_share(c_share) {}

  template <typename DerivedX, typename DerivedY>
  auto Multiply(const Eigen::MatrixBase<DerivedX>& x_share,
                const Eigen::MatrixBase<DerivedY>& y_share, IO<> io) {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto [d_share_matrix, d_share] = FlatbuffersComMatrix(x_share.rows(),
                                                          x_share.cols(),
                                                          builder);
    auto [e_share_matrix, e_share] = FlatbuffersComMatrix(y_share.rows(),
                                                          y_share.cols(),
                                                          builder);
  
    e_share = y_share - b_share;
    d_share = x_share - a_share;
  
    // Send our shares of d and e to the other party
    auto mt_inference_share = CreateMTInferenceShare(builder, d_share_matrix,
                                                     e_share_matrix);
    auto mt_inference_share_message = CreateServerMessage(
        builder, ServerMessageUnion_MTInferenceShare,
        mt_inference_share.Union());
    builder.Finish(mt_inference_share_message);
    io.Send(builder.Release());
    
    // The other party should send us their d and e shares
    auto their_de_shares = GetClientMessage(io.Receive().data())->message_as_MTInferenceShare();
   
    auto their_d_share_matrix = AsComMatrix(their_de_shares->dShare());
    auto their_e_share_matrix = AsComMatrix(their_de_shares->eShare());
  
    // Reconstruct d and e
    auto d = d_share + their_d_share_matrix;
    auto e = e_share + their_e_share_matrix;
    
    return AdjustMultiplication(d * e + d * b_share + e * a_share) + c_share;
  }
};

using MTProvider = std::function<
    MultiplicationTriplet(std::size_t, std::size_t, std::size_t)>;

inline auto GetMT(std::size_t m, std::size_t n, std::size_t k) {
  return MultiplicationTriplet {
    Eigen::MatrixX<Com>::Zero(m, n),
    Eigen::MatrixX<Com>::Zero(n, k),
    Eigen::MatrixX<Com>::Zero(m, k)
  };
}

}  // namespace amr

#endif  // ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_
