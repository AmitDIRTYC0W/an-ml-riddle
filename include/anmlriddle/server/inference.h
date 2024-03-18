// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_SERVER_INFERENCE_H_
#define ANMLRIDDLE_SERVER_INFERENCE_H_

#include <functional>
#include <future>
#include <optional>
#include <thread>

#include <flatbuffers/flatbuffers.h>

#include "server_message_generated.h"
#include <anmlriddle/com.h>
#include <anmlriddle/multiplication_triplet.h>

namespace anmlriddle {

namespace server {

class Inference {  
 private:
  std::function<void(const flatbuffers::DetachedBuffer&)> send_; // TODO Merge this kind of stuff with client's Inference (create a superclass)
  std::promise<void> result_;
  bool began_ = false;

  ModelShareT model_share_;
  flatbuffers::DetachedBuffer client_model_share_buf_;

  std::shared_ptr<SynchronisedQueue<>> layer_messages_;
  IO<> layer_io_;

  std::optional<std::vector<std::byte>> input_;  // NOTE Maybe use a pointer instead?
  std::mutex input_mutex_;
  std::condition_variable input_condition_;

  std::jthread layers_inference_;
  
  //SynchronisedQueue<???> mt_generation_messages_;
  //SynchronisedQueue<MultiplicationTriplet> multiplication_triplets_;
  //std::jthread mt_generation_;

  void InferLayers() noexcept;
  //void GenerateMTs() noexcept;

 public:
  Inference(std::function<void(const flatbuffers::DetachedBuffer&)> send,
            std::span<const std::byte> model_buf);
  Inference(Inference&) = delete;

  std::future<void> Begin();

  void Receive(const std::vector<std::byte> message);
};

}  // namespace server

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_SERVER_INFERENCE_H_
