// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_INFERENCE_H_
#define ANMLRIDDLE_CLIENT_INFERENCE_H_

#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <mutex>

#include <capnp/message.h>

namespace amrc {

class Inference {
 private:
  std::function<void(const ::capnp::MessageBuilder&)> send_;
  const std::vector<float> input_;
  std::promise<std::vector<float>> result_;

  std::vector<Layer> layers_;
  std::atomic_bool received_model_ = false;

  std::queue<const ::capnp::MessageReader&> layer_messages_;
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

  inline std::future<std::vector<float>>& get_result() {
    return result_.get_future();
  }
}

}  // namespace amrc

#endif  // ANMLRIDDLE_CLIENT_INFERENCE_H_
