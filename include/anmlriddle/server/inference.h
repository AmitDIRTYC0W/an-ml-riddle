// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_SERVER_INFERENCE_H_
#define ANMLRIDDLE_SERVER_INFERENCE_H_

#include <future>

#include <flatbuffers/flatbuffers.h>

#include "anmlriddle/channel.h"
#include "../../../src/server_message_generated.h"
#include <anmlriddle/com.h>
#include <anmlriddle/multiplication_triplet.h>

namespace anmlriddle {

namespace server {

class Inference {  
 public:
  explicit Inference(SendFunction send) : send_(send) {}
  Inference(Inference&) = delete;

  void Infer(std::span<const std::byte> model_buffer);

  void Receive(const std::vector<std::byte> message_buffer);

 private:
  SendFunction send_; // TODO Merge this kind of stuff with client's Inference (create a superclass)
  std::promise<void> result_;

  ModelShareT model_share_;
  flatbuffers::DetachedBuffer client_model_share_buf_;

  SingularSink<Dense> input_share_;
  SingularSink<MTInferenceShare> mt_inference_share_;
  
  //SynchronisedQueue<???> mt_generation_messages_;
  //SynchronisedQueue<MultiplicationTriplet> multiplication_triplets_;
  //std::jthread mt_generation_;

  void InferLayers(std::stop_token stop_token) noexcept;
  //void GenerateMTs() noexcept;
};

}  // namespace server

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_SERVER_INFERENCE_H_
