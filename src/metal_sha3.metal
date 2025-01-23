#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

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

kernel void sha3(device ulong *buffer [[ buffer(0) ]],
                 device ulong *output_1 [[ buffer(1) ]],
                 device ulong& nonce_start [[ buffer(2) ]],
                 device ulong& difficulty [[ buffer(3) ]],
                 device uint& num_rounds [[ buffer(4) ]],
                 uint gid [[ thread_position_in_grid ]],
                 uint threads_per_grid [[ threads_per_grid ]]
                 ) {

    ulong state[25];
    for (uint i = 0; i < num_rounds; i++) {
        for (uint j = 0; j < 25; j++) {
            state[j] = 0;
        }
        state[0] = nonce_start + gid + i * threads_per_grid;
        state[1] = buffer[1];
        state[2] = buffer[2];
        state[3] = buffer[3];
        state[4] = buffer[4];
        state[5] = buffer[5];

        state[16] ^= 0x8000000000000000ull;

        uint r, x, y, t;
        ulong tmp, current, C[5];
        for (r = 0; r < 24; ++r) {
            for (x = 0; x < 5; ++x) {
                C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }
            for (x = 0; x < 5; ++x) {
                tmp = C[(x + 4) % 5] ^ rotate(C[(x + 1) % 5], 1ull);
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
                for (x = 0; x < 5; ++x) {
                    C[x] = state[y + x];
                }
                for (x = 0; x < 5; ++x) {
                    state[x + y] = C[x] ^ (~C[(x + 1) % 5] & C[(x + 2) % 5]);
                }
            }
            state[0] ^= RC[r];
        }

        // Re-initialize state for rounds 2 and 3
        for (uint j = 4; j < 25; j++) {
            state[j] = 0;
        }
        state[4] = 0x06;
        state[16] = 0x8000000000000000ull;

        for (r = 0; r < 24; ++r) {
            for (x = 0; x < 5; ++x) {
                C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }
            for (x = 0; x < 5; ++x) {
                tmp = C[(x + 4) % 5] ^ rotate(C[(x + 1) % 5], 1ull);
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
                for (x = 0; x < 5; ++x) {
                    C[x] = state[y + x];
                }
                for (x = 0; x < 5; ++x) {
                    state[x + y] = C[x] ^ (~C[(x + 1) % 5] & C[(x + 2) % 5]);
                }
            }
            state[0] ^= RC[r];
        }

        // Re-initialize state for round 3
        for (uint j = 4; j < 25; j++) {
            state[j] = 0;
        }
        state[4] = 0x06;
        state[16] = 0x8000000000000000ull;

        // Round 3
        for (r = 0; r < 24; ++r) {
            for (x = 0; x < 5; ++x) {
                C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }
            for (x = 0; x < 5; ++x) {
                tmp = C[(x + 4) % 5] ^ rotate(C[(x + 1) % 5], 1ull);
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
                for (x = 0; x < 5; ++x) {
                    C[x] = state[y + x];
                }
                for (x = 0; x < 5; ++x) {
                    state[x + y] = C[x] ^ (~C[(x + 1) % 5] & C[(x + 2) % 5]);
                }
            }
            state[0] ^= RC[r];
        }

        // Check difficulty
        ulong swap = reverse_bits(state[0]);
        __atomic_thread_fence(memory_order::memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (swap < difficulty) {
            if (output_1[1] == 0 || output_1[1] > swap) {
                output_1[0] = nonce_start + gid + i * threads_per_grid;
                output_1[1] = swap;
            }
        } else {
            if (output_1[1] == 0 || output_1[1] > swap) {
                output_1[1] = swap;
            }
        }
    }
}
