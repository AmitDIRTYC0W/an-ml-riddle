// Copyright 2023 Amit Goren

#include "generate_shares.h"

#include <sodium.h>

void GenerateShares(const ComVec secret,
                    std::pair<ComVec&, ComList::Builder&> shares) {
  // Fill our share in random bytes
  shares.first = ComVec(secret.size());
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));

  for (std::size_t i = 0; i < secret.size(); ++i) {
    shares.second.set(i, secret[i] - shares.first[i]);
  }
}

