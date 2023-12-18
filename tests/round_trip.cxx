#include <iostream>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <amrc/inference.h>

void PrintMessage(::capnp::MessageBuilder& message) {
  ::capnp::writeMessageToFd(1, message);
  // auto arr = ::capnp::messageToFlatArray(message);
  // std::cout << arr << std::endl;
  // KJ_DBG(arr);
}

int main() {
  amrc::Inference inference = amrc::Inference(PrintMessage);
  inference.Begin({1.0, 2.0, 3.0});
  // inference.Receive(NULL);
  // int no = decrypt(encrypt(3));
  // return EXIT_SUCCESS;
}
