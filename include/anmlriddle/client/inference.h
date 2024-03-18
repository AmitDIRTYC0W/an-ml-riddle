// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INFERENCE_H_
#define ANMLRIDDLE_CLIENT_INFERENCE_H_

#include <vector>
#include <future>

#include <Eigen/Dense>

#include <anmlriddle/channel.h>
#include "../../../src/common_generated.h"
#include "../../../src/server_message_generated.h"
#include <anmlriddle/com.h>

namespace anmlriddle {

namespace client {

class Inference {
 public:
  explicit Inference(SendFunction send) : send_(send) {}
  Inference(Inference&) = delete;

  std::vector<float> Infer(std::span<float> input);

  void Receive(std::vector<std::byte>& message_buffer);
 
 private:
  void InferLayers(std::stop_token stop_token, std::span<float> input) noexcept;

  SendFunction send_;
  
  std::promise<std::vector<float>> output_promise_;
  
  SingularSink<ModelShare> model_share_;
  SingularSink<Dense> their_output_share_;
  SingularSink<MTInferenceShare> mt_inference_share_;
};

}  // namespace client

}  // namespace anmlriddle





















// namespace amrc {

// class Layer;

// class Inference {
//   friend class Layer;
  
//  private:
//   std::function<void(const ::capnp::MessageBuilder&)> send_;
//   const std::vector<float> input_;
//   std::promise<std::vector<float>> result_;
//   bool began_ = false;

//   std::vector<std::unique_ptr<Layer>> layers_;

//   std::queue<ServerMessage::Reader> layer_messages_;
//   std::mutex layer_messages_mutex_;
//   std::condition_variable layer_messages_condition_;

//   std::optional<ServerMessage::Reader> model_share_;
//   std::mutex model_share_mutex_;
//   std::condition_variable model_share_condition_;

//   std::jthread layers_inference_;

//   void FetchLayers();
//   void InferLayers() noexcept;

//   ComVec ShareInput();

//  public:
//   Inference(std::function<void(const ::capnp::MessageBuilder&)> send,
//             std::vector<float> input) : send_(send), input_(input) {}
//   Inference(Inference&) = delete;

//   void Begin(); // TODO maybe noexcept?

//   void Receive(::capnp::MessageReader& message);

//   inline std::future<std::vector<float>> get_result() noexcept {
//     return result_.get_future();
//   }
// };

// }  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_INFERENCE_H_
