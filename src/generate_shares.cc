// Copyright 2023 Amit Goren

#include "generate_shares.h"

#include <algorithm>
#include <execution>
#include <ranges>

#include <sodium.h>

void GenerateShares(const ComVec secret,
                    std::pair<ComVec&, ComList::Builder&> shares) {
  // Fill our share in random bytes
  shares.first = ComVec(secret.size());
  randombytes_buf(shares.first.data(), shares.first.size() * sizeof(Com));

  // other share = secret - our share
  // FIXME use par_unseq
  std::ranges::for_each(std::views::iota((ComVec::size_type)0, secret.size()),
    [&](unsigned long i) {shares.second.set(i, secret[i] - shares.first[i]);});
}

