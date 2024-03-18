// Copyright 2024 Amit Goren

#ifndef ANMLRIDDLE_CHANNEL_H_
#define ANMLRIDDLE_CHANNEL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>

namespace anmlriddle {

using SendFunction = std::function<void(const flatbuffers::DetachedBuffer&)>;

template <typename T>
class MessagesSink {
 public:
  void Insert(std::vector<std::byte>& message_buffer, T* message) {
    {
      std::scoped_lock<std::mutex> lock(mutex_);
      std::unique_ptr pointer(message, &Delete);  // NOTE we might need to change this to shared_ptr (and then maybe use std::owner_less)
      message_buffers_[pointer] = std::move(message_buffer);
    }
    cv_.notify_one();
  }
  
  std::unique_ptr<T> Read(std::stop_token stop_token) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, stop_token, [&] { return !message_buffers_.empty(); });
    return std::move(message_buffers_.cbegin()->first);
  }
 
 private:
  std::unordered_map<std::unique_ptr<T>, std::vector<std::byte>> message_buffers_;
  std::mutex mutex_;
  std::condition_variable_any cv_;
  
  void Delete(T* message) {
    std::scoped_lock<std::mutex> lock(mutex_);
    message_buffers_.erase(std::unique_ptr(message));
  }
};

template <typename T>
class SingularSink {
 public:
  void Insert(std::vector<std::byte>& message_buffer, const T* message) {
    {
      std::scoped_lock<std::shared_mutex> lock(mutex_);
      message_buffer_ = std::move(message_buffer);
      message_ = message;
    }
    cv_.notify_all();
  }

  const T* Read() {
    std::shared_lock lock(mutex_);
    cv_.wait(lock, [&] { return message_buffer_.has_value(); });
    return message_;
  }
  
  // WARNING This may return nullptr
  const T* Read(std::stop_token stop_token) {
    std::shared_lock lock(mutex_);
    bool success = cv_.wait(lock, stop_token, [&] { return message_buffer_.has_value(); });
    return success ? message_ : nullptr;
  }
  
 private:
  std::optional<std::vector<std::byte>> message_buffer_;
  const T* message_;
  std::shared_mutex mutex_;
  std::condition_variable_any cv_;
};

}  // namespace anmlriddle

#endif  // ANMLRIDDLE_CHANNEL_H_