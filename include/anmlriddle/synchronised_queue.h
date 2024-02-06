// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_SYNCHRONISED_QUEUE_H_
#define ANMLRIDDLE_SYNCHRONISED_QUEUE_H_

#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <memory>

#include <flatbuffers/flatbuffers.h>

namespace amr {

template<typename T = std::vector<std::byte>>
struct SynchronisedQueue {
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !queue_.empty(); });

    T element = queue_.front();
    queue_.pop();
    return element;
  }

  void Push(T element) {
    {
	    std::unique_lock<std::mutex> lock(mutex_);
	    queue_.push(element);
    }
    cv_.notify_one();
  }
};

template<typename I = std::vector<std::byte>, typename O = flatbuffers::DetachedBuffer>
class IO {
 private:
  std::shared_ptr<SynchronisedQueue<I>> input_;
  std::function<void(const O&)> send_;

 public:
   IO(std::shared_ptr<SynchronisedQueue<I>> input, std::function<void(const O&)> send)
       : input_(input), send_(send) {}

  void Send(const O& message) {
    send_(message);
  }

  I Receive() {
    return input_->Pop();
  }
};

  /*
template <typename M, typename T>
concept IO = requires(T io) {
	{io.Send(M message)};
	{io.Receive()} -> std::convertible_to<M>;
};
*/

}  // namespace amr

#endif  // ANMLRIDDLE_SYNCHRONISED_QUEUE_H_
