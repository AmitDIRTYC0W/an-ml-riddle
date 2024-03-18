// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_
#define ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_

#include <Eigen/Dense>
#include <stop_token>

#include "channel.h"
#include "com.h"
#include "server_message_generated.h"
#include "common_generated.h"

namespace anmlriddle {

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
                const Eigen::MatrixBase<DerivedY>& y_share, SendFunction send,
                SingularSink<MTInferenceShare>& messaegesSink, std::stop_token stop_token) {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto [d_share_matrix, d_share] = FlatbuffersMatrix(x_share.rows(),
                                                          x_share.cols(),
                                                          builder);
    auto [e_share_matrix, e_share] = FlatbuffersMatrix(y_share.rows(),
                                                          y_share.cols(),
                                                          builder);
  
    e_share = y_share - b_share;
    d_share = x_share - a_share;
  
    // Send our shares of d and e to the other party
    // FIXME TODO XXX there should be seperate versions for server and client!
    auto mt_inference_share = CreateMTInferenceShare(builder, d_share_matrix,
                                                     e_share_matrix);
    auto mt_inference_share_message = CreateServerMessage(
        builder, ServerMessageUnion_MTInferenceShare,
        mt_inference_share.Union());
    builder.Finish(mt_inference_share_message);
    send(builder.Release());
    
    // The other party should send us their d and e shares
    auto their_de_shares = messaegesSink.Read(stop_token);
   
    auto their_d_share_matrix = AsEigenMatrix(their_de_shares->dShare());
    auto their_e_share_matrix = AsEigenMatrix(their_de_shares->eShare());
  
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

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_MULTIPLICATION_TRIPLET_H_
