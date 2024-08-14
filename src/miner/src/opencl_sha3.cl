constant static const ulong rot[24] = {1,  3,  6,  10, 15, 21, 28, 36,
                                       45, 55, 2,  14, 27, 41, 56, 8,
                                       25, 43, 62, 18, 39, 61, 20, 44};

constant static const int pos[24] = {10, 7,  11, 17, 18, 3,  5,  16,
                                     8,  21, 24, 4,  15, 23, 19, 13,
                                     12, 2,  20, 14, 22, 9,  6,  1};

constant static const ulong RC[] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808aul,
    0x8000000080008000ul, 0x000000000000808bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000aul,
    0x000000008000808bul, 0x800000000000008bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800aul, 0x800000008000000aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

kernel void sha3(global uchar *buffer, uint size, ulong timestamp,
                 ulong nonce_offset, global ulong *difficulty,
                 global uchar *target_difficulty) {
  // The padding of the buffer outside of the gpu, so it doesn't have to run on
  // every gpu thread. This is the padding for the second run.
  uchar output_buffer[136] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128};

  ulong state[25] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // uint w = 64;
  uint blockSizeBytes = 17;
  uint blocksizeBits = 136;
  for (uint i = 0; i < size; i += blocksizeBits) {
    for (uint j = 0; j < blockSizeBytes; ++j) {
      ulong v = convert_ulong(buffer[i + j * 8]) +
                (convert_ulong(buffer[i + j * 8 + 1]) << 8) +
                (convert_ulong(buffer[i + j * 8 + 2]) << 16) +
                (convert_ulong(buffer[i + j * 8 + 3]) << 24) +
                (convert_ulong(buffer[i + j * 8 + 4]) << 32) +
                (convert_ulong(buffer[i + j * 8 + 5]) << 40) +
                (convert_ulong(buffer[i + j * 8 + 6]) << 48) +
                (convert_ulong(buffer[i + j * 8 + 7]) << 56);
      if (i + j * 8 == 0)
        v = nonce_offset + get_global_id(0);
      if (i + j * 8 == 8)
        v = timestamp;
      state[j] ^= v;
    }
    uint r, x, y, t;
    ulong tmp, current, C[5];
    for (r = 0; r < 24; ++r) {
      for (x = 0; x < 5; ++x) {
        C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^
               state[x + 20];
      }
      for (x = 0; x < 5; ++x) {
        tmp = C[(x + 4) % 5] ^ rotate(C[(x + 1) % 5], 1ul);
        for (y = 0; y < 5; ++y) {
          state[x + y * 5] ^= tmp;
        }
      }
      current = state[1];
      for (t = 0; t < 24; ++t) {
        tmp = state[pos[t]];
        state[pos[t]] = rotate(current, rot[t]);
        current = tmp;
      }
      for (y = 0; y < 25; y += 5) {
        for (x = 0; x < 5; ++x)
          C[x] = state[y + x];
        for (x = 0; x < 5; ++x) {
          state[x + y] = C[x] ^ (~C[(x + 1) % 5] & C[(x + 2) % 5]);
        }
      }
      state[0] ^= RC[r];
    }
  }
  for (uint i = 0; i < 32; ++i) {
    output_buffer[i] = (state[i / 8] >> ((i % 8) * 8));
  }
  // 2nd sha3
  for (uint i = 0; i < 25; ++i) {
    state[i] = 0;
  }
  for (uint j = 0; j < blockSizeBytes; ++j) {
    ulong v = convert_ulong(output_buffer[j * 8]) +
              (convert_ulong(output_buffer[j * 8 + 1]) << 8) +
              (convert_ulong(output_buffer[j * 8 + 2]) << 16) +
              (convert_ulong(output_buffer[j * 8 + 3]) << 24) +
              (convert_ulong(output_buffer[j * 8 + 4]) << 32) +
              (convert_ulong(output_buffer[j * 8 + 5]) << 40) +
              (convert_ulong(output_buffer[j * 8 + 6]) << 48) +
              (convert_ulong(output_buffer[j * 8 + 7]) << 56);
    state[j] ^= v;
  }
  uint r, x, y, t;
  ulong tmp, current, C[5];
  for (r = 0; r < 24; ++r) {
    for (x = 0; x < 5; ++x) {
      C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^
             state[x + 20];
    }
    for (x = 0; x < 5; ++x) {
      tmp = C[(x + 4) % 5] ^ rotate(C[(x + 1) % 5], 1ul);
      for (y = 0; y < 5; ++y) {
        state[x + y * 5] ^= tmp;
      }
    }
    current = state[1];
    for (t = 0; t < 24; ++t) {
      tmp = state[pos[t]];
      state[pos[t]] = rotate(current, rot[t]);
      current = tmp;
    }
    for (y = 0; y < 25; y += 5) {
      for (x = 0; x < 5; ++x)
        C[x] = state[y + x];
      for (x = 0; x < 5; ++x) {
        state[x + y] = C[x] ^ (~C[(x + 1) % 5] & C[(x + 2) % 5]);
      }
    }
    state[0] ^= RC[r];
  }
  for (uint i = 0; i < 32; ++i) {
    output_buffer[i] = (state[i / 8] >> ((i % 8) * 8));
  }

  // Compare difficulty
  bool le = true;
  for (uint i = 0; i < 32; ++i) {
    if (output_buffer[i] < target_difficulty[i])
      break;
    if (output_buffer[i] > target_difficulty[i])
      le = false;
  }
  if (le) {
    difficulty[0] = nonce_offset + get_global_id(0);
    // 256bit number representation
    uchar n[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ulong q = 0;
    for (int i = 255; i >= 0; --i) {
      int r = 0;
      // n <<= 1
      for (int j = 31; j >= 0; --j) {
        if (n[j] >= 128) {
          n[j] -= 128;
          n[j] <<= 1;
          n[j] += r;
          r = 1;
        } else {
          n[j] <<= 1;
          n[j] += r;
          r = 0;
        }
      }
      // n += 1
      n[31] += 1;
      // if n >= d
      bool ge = true;
      for (int j = 0; j < 32; ++j) {
        if (n[j] < output_buffer[j]) {
          ge = false;
          break;
        }
        if (n[j] > output_buffer[j]) {
          break;
        }
      }
      if (ge) {
        // n -= d
        int r = 0;
        for (int j = 31; j >= 0; --j) {
          // There is no temporary overflow, because in OpenCL uchar + uchar is
          // ulong (not really sure, but it's bigger than uchar)
          if (n[j] < output_buffer[j] + r) {
            n[j] = n[j] - r - output_buffer[j];
            r = 1;
          } else {
            n[j] = n[j] - r - output_buffer[j];
            r = 0;
          }
        }
        // set the bits just for low 64bits
        if (i < 64) {
          q += 1ul << i;
        }
      }
    }
    difficulty[1] = q;
  }
}