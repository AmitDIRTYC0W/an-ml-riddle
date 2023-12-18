// Copyright 2023 Amit Goren

#ifndef ANMLRIDDLE_GENERATE_SHARES_H_
#define ANMLRIDDLE_GENERATE_SHARES_H_

#include <utility>

#include <anmlriddle/com.h>

void GenerateShares(const ComVec secret,
                    std::pair<ComVec&, ComList::Builder&> shares);

#endif  // ANMLRIDDLE_GENERATE_SHARES_H_
