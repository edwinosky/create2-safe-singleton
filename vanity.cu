// vanity.cu
// Compilar:
// nvcc -arch=sm_61 -O3 vanity.cu -o vanity

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <cuda_runtime.h>

// ------------------ Host helpers ------------------
static inline int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int hexToBytes(const char *hex, uint8_t *out, size_t outlen) {
    size_t hexlen = strlen(hex);
    if (hexlen >= 2 && hex[0]=='0' && (hex[1]=='x' || hex[1]=='X')) {
        hex += 2; hexlen -= 2;
    }
    if (hexlen != outlen*2) return -1;
    for (size_t i = 0; i < outlen; ++i) {
        int hi = hexVal(hex[2*i]);
        int lo = hexVal(hex[2*i+1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (hi << 4) | lo;
    }
    return 0;
}

void printHex(const uint8_t *b, size_t n) {
    for (size_t i=0;i<n;i++) printf("%02x", b[i]);
    printf("\n");
}

// ------------------ Keccak-f[1600] / Keccak-256 device-side ------------------
// Minimal correct implementation suited for device use.

__device__ inline uint64_t rol64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ void keccakf(uint64_t st[25]) {
    const uint64_t rc[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
        0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
        0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
    };
    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x=0;x<5;x++) C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        uint64_t D[5];
        for (int x=0;x<5;x++) D[x] = C[(x+4)%5] ^ rol64(C[(x+1)%5], 1);
        for (int x=0;x<5;x++) for (int y=0;y<5;y++) st[x + 5*y] ^= D[x];

        // manual rho/pi mapping (compact)
        const int r[25] = {
             0,  1, 62, 28, 27,
            36, 44,  6, 55, 20,
             3,10, 43, 25, 39,
            41,45, 15, 21,  8,
            18, 2, 61, 56, 14
        };
        uint64_t B[25];
        B[0] = st[0];
        B[1] = rol64(st[6], r[1]);
        B[2] = rol64(st[12], r[2]);
        B[3] = rol64(st[18], r[3]);
        B[4] = rol64(st[24], r[4]);

        B[5] = rol64(st[3], r[5]);
        B[6] = rol64(st[9], r[6]);
        B[7] = rol64(st[10], r[7]);
        B[8] = rol64(st[16], r[8]);
        B[9] = rol64(st[22], r[9]);

        B[10] = rol64(st[1], r[10]);
        B[11] = rol64(st[7], r[11]);
        B[12] = rol64(st[13], r[12]);
        B[13] = rol64(st[19], r[13]);
        B[14] = rol64(st[20], r[14]);

        B[15] = rol64(st[4], r[15]);
        B[16] = rol64(st[5], r[16]);
        B[17] = rol64(st[11], r[17]);
        B[18] = rol64(st[17], r[18]);
        B[19] = rol64(st[23], r[19]);

        B[20] = rol64(st[2], r[20]);
        B[21] = rol64(st[8], r[21]);
        B[22] = rol64(st[14], r[22]);
        B[23] = rol64(st[15], r[23]);
        B[24] = rol64(st[21], r[24%25]);

        for (int i=0;i<25;i++) st[i] = B[i] ^ ((~B[(i+5)%25]) & B[(i+10)%25]);

        st[0] ^= rc[round];
    }
}

__device__ void keccak256_device(const uint8_t *in, size_t inlen, uint8_t *out) {
    const int rate = 136;
    uint64_t st[25];
    for (int i=0;i<25;i++) st[i] = 0;

    size_t offset = 0;
    while (inlen - offset >= (size_t)rate) {
        for (int i=0;i < rate/8; ++i) {
            uint64_t t = 0;
            for (int j=0;j<8;j++) t |= ((uint64_t)in[offset + i*8 + j]) << (8*j);
            st[i] ^= t;
        }
        keccakf(st);
        offset += rate;
    }
    uint8_t block[136];
    int rem = inlen - offset;
    for (int i=0;i<rate;i++) block[i] = 0;
    for (int i=0;i<rem;i++) block[i] = in[offset + i];
    block[rem] = 0x06;
    block[rate-1] |= 0x80;

    for (int i=0;i<rate/8;i++) {
        uint64_t t = 0;
        for (int j=0;j<8;j++) t |= ((uint64_t)block[i*8 + j]) << (8*j);
        st[i] ^= t;
    }
    keccakf(st);

    for (int i=0;i<4;i++) {
        uint64_t w = st[i];
        for (int j=0;j<8;j++) out[i*8 + j] = (uint8_t)((w >> (8*j)) & 0xff);
    }
}

// ------------------ device string / hex helpers ------------------
__device__ int d_strlen(const char *s) {
    int i=0; while (s[i] != 0) i++; return i;
}
__device__ int d_strncmp(const char *a, const char *b, int n) {
    for (int i=0;i<n;i++) {
        unsigned char ca = a[i], cb = b[i];
        if (ca != cb) return (int)ca - (int)cb;
        if (ca == 0) return 0;
    }
    return 0;
}
__device__ void byteToHex(uint8_t v, char *out) {
    const char map[] = "0123456789abcdef";
    out[0] = map[(v >> 4) & 0xF];
    out[1] = map[v & 0xF];
}

// ------------------ Kernel (inner loop enabled) ------------------
extern "C"
__global__ void searchKernel(
    const uint8_t *d_factory,      // 20 bytes
    const uint8_t *d_initHash,     // 32 bytes
    uint64_t globalStartSalt,
    uint64_t batchSize,
    const char *d_prefix, int prefixLen,
    const char *d_suffix, int suffixLen,
    unsigned int *d_foundFlag,          // 0/1 flag (unsigned int)
    unsigned long long *d_foundSalt,
    unsigned char *d_foundAddr, // 40 bytes hex ASCII
    uint64_t innerIter   // salts per thread
) {
    uint64_t totalThreads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    // thread-local small buffers
    uint8_t buf[1 + 20 + 32 + 32];
    buf[0] = 0xff;
    for (int i=0;i<20;i++) buf[1+i] = d_factory[i];

    for (uint64_t iter = 0; iter < innerIter; ++iter) {
        // early exit if someone else found
        if (atomicAdd(d_foundFlag, 0u) != 0u) return;

        uint64_t salt = globalStartSalt + tid + iter * totalThreads;
        if (salt >= globalStartSalt + batchSize) return;

        // salt bytes32 (we place low 8 bytes as value; rest zero)
        for (int i=0;i<32;i++) buf[21 + i] = 0;
        uint64_t tmp = salt;
        for (int i=0;i<8;i++) buf[21 + 31 - i] = (uint8_t)((tmp >> (8*i)) & 0xff);

        // initHash copy
        for (int i=0;i<32;i++) buf[53 + i] = d_initHash[i];

        // keccak
        uint8_t hash[32];
        keccak256_device(buf, sizeof(buf), hash);

        // address = last 20 bytes hash[12..31] -> hex (no 0x)
        char addrHex[41];
        for (int i=0;i<20;i++) byteToHex(hash[12 + i], &addrHex[i*2]);
        addrHex[40] = 0;

        // prefix/suffix compare
        bool ok = true;
        if (prefixLen > 0) {
            if (d_strncmp(addrHex, d_prefix, prefixLen) != 0) ok = false;
        }
        if (ok && suffixLen > 0) {
            if (d_strncmp(&addrHex[40 - suffixLen], d_suffix, suffixLen) != 0) ok = false;
        }

        if (ok) {
            // claim found (first writer wins)
            unsigned int old = atomicExch(d_foundFlag, 1u);
            if (old == 0u) {
                *d_foundSalt = salt;
                for (int i=0;i<40;i++) d_foundAddr[i] = (unsigned char)addrHex[i];
            }
            return;
        }
    } // innerIter
}

// ------------------ Host launcher / CLI ------------------
void usage(const char *n) {
    printf("Usage: %s --factory 0x... --initcodehash 0x... [--prefix hex] [--suffix hex] [--start N] [--batch N] [--blocks N] [--threads N] [--inner N]\n", n);
    printf(" Example: %s --factory 0xC0DE... --initcodehash 0xabcdef... --prefix 7702 --suffix 7702 --start 0 --batch 1000000000 --blocks 256 --threads 256 --inner 10000\n", n);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }

    const char *factoryHex = NULL;
    const char *initHashHex = NULL;
    const char *prefix = "";
    const char *suffix = "";
    uint64_t start = 0;
    uint64_t batch = 1000000;
    int blocks = 128;
    int threads = 128;
    uint64_t inner = 1;

    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i],"--factory")==0 && i+1<argc) factoryHex = argv[++i];
        else if (strcmp(argv[i],"--initcodehash")==0 && i+1<argc) initHashHex = argv[++i];
        else if (strcmp(argv[i],"--prefix")==0 && i+1<argc) prefix = argv[++i];
        else if (strcmp(argv[i],"--suffix")==0 && i+1<argc) suffix = argv[++i];
        else if (strcmp(argv[i],"--start")==0 && i+1<argc) start = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"--batch")==0 && i+1<argc) batch = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"--blocks")==0 && i+1<argc) blocks = atoi(argv[++i]);
        else if (strcmp(argv[i],"--threads")==0 && i+1<argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i],"--inner")==0 && i+1<argc) inner = strtoull(argv[++i], NULL, 10);
    }

    if (!factoryHex || !initHashHex) { usage(argv[0]); return 1; }

    char prefixClean[128]; prefixClean[0]=0;
    char suffixClean[128]; suffixClean[0]=0;
    strncpy(prefixClean, prefix, sizeof(prefixClean)-1);
    strncpy(suffixClean, suffix, sizeof(suffixClean)-1);
    int prefixLen = strlen(prefixClean);
    int suffixLen = strlen(suffixClean);

    uint8_t factory[20];
    uint8_t initHash[32];
    if (hexToBytes(factoryHex, factory, 20) != 0) {
        fprintf(stderr, "Invalid factory hex (expect 40 hex chars + optional 0x)\n"); return 1;
    }
    if (hexToBytes(initHashHex, initHash, 32) != 0) {
        fprintf(stderr, "Invalid initcodehash hex (expect 64 hex chars + optional 0x)\n"); return 1;
    }

    // device allocations
    uint8_t *d_factory;
    uint8_t *d_initHash;
    char *d_prefix;
    char *d_suffix;
    unsigned int *d_foundFlag;
    unsigned long long *d_foundSalt;
    unsigned char *d_foundAddr;

    cudaMalloc(&d_factory, 20);
    cudaMalloc(&d_initHash, 32);
    cudaMalloc(&d_prefix, 128);
    cudaMalloc(&d_suffix, 128);
    cudaMalloc(&d_foundFlag, sizeof(unsigned int));
    cudaMalloc(&d_foundSalt, sizeof(unsigned long long));
    cudaMalloc(&d_foundAddr, 40);

    cudaMemcpy(d_factory, factory, 20, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initHash, initHash, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_foundFlag, 0, sizeof(unsigned int));
    cudaMemset(d_foundSalt, 0, sizeof(unsigned long long));
    cudaMemset(d_foundAddr, 0, 40);

    char prefixBuf[128]; memset(prefixBuf,0,sizeof(prefixBuf));
    char suffixBuf[128]; memset(suffixBuf,0,sizeof(suffixBuf));
    for (int i=0;i<prefixLen;i++) {
        char c = prefixClean[i];
        if (c >= 'A' && c <= 'F') c = c - 'A' + 'a';
        prefixBuf[i] = c;
    }
    for (int i=0;i<suffixLen;i++) {
        char c = suffixClean[i];
        if (c >= 'A' && c <= 'F') c = c - 'A' + 'a';
        suffixBuf[i] = c;
    }
    cudaMemcpy(d_prefix, prefixBuf, 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix, suffixBuf, 128, cudaMemcpyHostToDevice);

    printf("CUDA Vanity search (demo)\n");
    printf("Factory: 0x"); printHex(factory,20);
    printf("InitCodeHash: 0x"); printHex(initHash,32);
    printf("Prefix: %s  Suffix: %s  Start: %llu  Batch: %llu  Blocks: %d Threads: %d Inner: %llu\n",
           prefixBuf, suffixBuf, (unsigned long long)start, (unsigned long long)batch, blocks, threads, (unsigned long long)inner);
    fflush(stdout);

    dim3 grid(blocks), block(threads);
    unsigned long long totalThreads = (unsigned long long)blocks * (unsigned long long)threads;
    if (totalThreads == 0) { fprintf(stderr,"bad grid\n"); return 1; }

    // Launch single (large) kernel that does innerIter per thread
    searchKernel<<<grid, block>>>(
        d_factory, d_initHash,
        (unsigned long long)start,
        (unsigned long long)batch,
        d_prefix, prefixLen,
        d_suffix, suffixLen,
        d_foundFlag,
        d_foundSalt,
        d_foundAddr,
        (unsigned long long)inner
    );

    cudaDeviceSynchronize();

    unsigned int foundFlagHost = 0;
    cudaMemcpy(&foundFlagHost, d_foundFlag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (foundFlagHost != 0) {
        unsigned long long foundSaltHost = 0;
        char addrBuf[41]; memset(addrBuf,0,41);
        cudaMemcpy(&foundSaltHost, d_foundSalt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(addrBuf, d_foundAddr, 40, cudaMemcpyDeviceToHost);
        printf("FOUND! salt=%llu (0x%016llx) address=0x%.*s\n",
               (unsigned long long)foundSaltHost, (unsigned long long)foundSaltHost, 40, addrBuf);
    } else {
        printf("No match found in this batch (tested up to %llu salts)\n", (unsigned long long)batch);
    }

    // cleanup
    cudaFree(d_factory); cudaFree(d_initHash); cudaFree(d_prefix);
    cudaFree(d_suffix); cudaFree(d_foundFlag); cudaFree(d_foundSalt); cudaFree(d_foundAddr);

    return 0;
}
