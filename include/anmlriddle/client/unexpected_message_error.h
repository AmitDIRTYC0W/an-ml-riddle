// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_CLIENT_UNEXPECTED_MESSAGE_ERROR_H_
#define ANMLRIDDLE_CLIENT_UNEXPECTED_MESSAGE_ERROR_H_

#include <stdexcept>

namespace amrc {
class UnexpectedMessageError : public std::logic_error {
 public:
    explicit UnexpectedMessageError(const char* what_arg)
        : std::logic_error(what_arg) {}
};
}  // amrc


#endif  // ANMLRIDDLE_CLIENT_UNEXPECTED_MESSAGE_ERROR_H_
