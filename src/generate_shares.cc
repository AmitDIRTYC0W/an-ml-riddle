// Copyright 2023 Amit Goren

#include "generate_shares.h"

#include <sodium.h>

void GenerateShares(const ComVec secret,
                    std::pair<ComVec&, ComList::Builder&> shares) {
  // Fill our share in random bytes
  shares.first = ComVec(secret.size());
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));

  // other share = secret - our share
  std::transform(std::par_unseq, secret.begin(), secret.end(), shares.first.begin(),
    shares.second.begin(), std::minus<Com>);
}

