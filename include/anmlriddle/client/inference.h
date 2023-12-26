// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INFERENCE_H_
#define ANMLRIDDLE_CLIENT_INFERENCE_H_

#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include <capnp/message.h>

#include <anmlriddle/client/layer.h>
#include "Model.capnp.h"

namespace amrc {

class Layer;

class Inference {
  friend class Layer;
  
 private:
  std::function<void(const ::capnp::MessageBuilder&)> send_;
  const std::vector<float> input_;
  std::promise<std::vector<float>> result_;

  std::vector<std::unique_ptr<Layer>> layers_;
  std::atomic_bool received_model_ = false;

  std::queue<ServerMessage::Reader> layer_messages_;
  std::mutex layer_messages_mutex_;
  std::condition_variable layer_messages_condition_;

  std::jthread layers_inference_;

  void SetupModel(const ModelShare::Reader& model_share);
  void StartOnlinePhase();
  void InferLayers();

 public:
  Inference(std::function<void(const ::capnp::MessageBuilder&)> send,
            std::vector<float> input) : send_(send), input_(input) {}
  Inference(Inference& other) = delete;

  void Receive(::capnp::MessageReader& message);

  inline std::future<std::vector<float>> get_result() {
    return result_.get_future();
  }
};

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_INFERENCE_H_
