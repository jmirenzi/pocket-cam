// imgproc.h - image pipeline for the thermal camera sweep.
// Plain C++11, no Arduino dependencies, so it can also be compiled and tested on a PC.
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "blue_noise.h"

namespace imgproc {

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int iclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------- enums / settings
enum Dither {
  D_THRESH, D_BAYER4, D_BAYER8, D_CLUSTER8, D_BLUE,
  D_FS, D_FS_SERP, D_ATKINSON, D_SIERRA_LITE, D_BURKES, D_STUCKI, D_JJN
};
enum Resize { R_NEAREST, R_BOX, R_BILINEAR };

inline const char *ditherName(Dither d) {
  switch (d) {
    case D_THRESH: return "THRESHOLD";
    case D_BAYER4: return "BAYER-4x4";
    case D_BAYER8: return "BAYER-8x8";
    case D_CLUSTER8: return "CLUSTERED-DOT-8x8";
    case D_BLUE: return "BLUE-NOISE-64";
    case D_FS: return "FLOYD-STEINBERG";
    case D_FS_SERP: return "FLOYD-STEINBERG-SERP";
    case D_ATKINSON: return "ATKINSON";
    case D_SIERRA_LITE: return "SIERRA-LITE";
    case D_BURKES: return "BURKES";
    case D_STUCKI: return "STUCKI";
    case D_JJN: return "JARVIS-JUDICE-NINKE";
  }
  return "?";
}
inline const char *resizeName(Resize r) {
  return r == R_NEAREST ? "nearest" : (r == R_BOX ? "box" : "bilinear");
}

// One complete set of pipeline settings. Plain aggregate on purpose (C++11).
struct Variant {
  const char *name;  // label printed above the strip
  Dither dither;
  Resize resize;
  uint8_t blur;      // 0 = none, 1 = 3x3 gaussian blur, 2 = 3x3 median
  float usm;         // unsharp-mask amount, 0 = off
  float clip;        // auto-levels clip fraction (0.02 = 2%), negative = skip levels
  bool eq;           // global histogram equalization
  bool clahe;        // local contrast (CLAHE), 4x4 tiles
  float gamma;       // <1 lightens midtones, >1 darkens
  bool avg4;         // use the 4-frame averaged capture
  uint8_t heat;      // ESC 7 heating time (printer darkness)
  uint8_t coarse;    // dot size: 1 = native, 2 = each dot printed as 2x2, ...
};

// ---------------------------------------------------------------- resize
// src is S x S, dst is W x H
inline void resizeImage(const uint8_t *src, int S, uint8_t *dst, int W, int H, Resize m) {
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int v;
      if (m == R_NEAREST) {
        int sx = ((2 * x + 1) * S) / (2 * W);
        int sy = ((2 * y + 1) * S) / (2 * H);
        v = src[sy * S + sx];
      } else if (m == R_BOX) {
        int sx0 = (x * S) / W, sx1 = imax(sx0 + 1, ((x + 1) * S) / W);
        int sy0 = (y * S) / H, sy1 = imax(sy0 + 1, ((y + 1) * S) / H);
        uint32_t sum = 0;
        for (int sy = sy0; sy < sy1; sy++)
          for (int sx = sx0; sx < sx1; sx++) sum += src[sy * S + sx];
        uint32_t n = (uint32_t)((sx1 - sx0) * (sy1 - sy0));
        v = (int)((sum + n / 2) / n);
      } else {
        float fx = (x + 0.5f) * S / W - 0.5f;
        float fy = (y + 0.5f) * S / H - 0.5f;
        int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
        float ax = fx - x0, ay = fy - y0;
        int x1 = iclamp(x0 + 1, 0, S - 1), y1 = iclamp(y0 + 1, 0, S - 1);
        x0 = iclamp(x0, 0, S - 1);
        y0 = iclamp(y0, 0, S - 1);
        float top = src[y0 * S + x0] * (1 - ax) + src[y0 * S + x1] * ax;
        float bot = src[y1 * S + x0] * (1 - ax) + src[y1 * S + x1] * ax;
        v = (int)(top * (1 - ay) + bot * ay + 0.5f);
      }
      dst[y * W + x] = (uint8_t)iclamp(v, 0, 255);
    }
  }
}

// ---------------------------------------------------------------- denoise / sharpen
// Separable [1 2 1]/4 blur. tmp must be W*H bytes.
inline void blur3(uint8_t *img, int W, int H, uint8_t *tmp) {
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      int xm = imax(x - 1, 0), xp = imin(x + 1, W - 1);
      tmp[y * W + x] = (uint8_t)((img[y * W + xm] + 2 * img[y * W + x] + img[y * W + xp] + 2) >> 2);
    }
  for (int y = 0; y < H; y++) {
    int ym = imax(y - 1, 0), yp = imin(y + 1, H - 1);
    for (int x = 0; x < W; x++)
      img[y * W + x] = (uint8_t)((tmp[ym * W + x] + 2 * tmp[y * W + x] + tmp[yp * W + x] + 2) >> 2);
  }
}

inline void median3(uint8_t *img, int W, int H, uint8_t *tmp) {
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      uint8_t v[9];
      int n = 0;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
          v[n++] = img[iclamp(y + dy, 0, H - 1) * W + iclamp(x + dx, 0, W - 1)];
      for (int i = 1; i < 9; i++) {  // insertion sort
        uint8_t k = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
        v[j + 1] = k;
      }
      tmp[y * W + x] = v[4];
    }
  memcpy(img, tmp, (size_t)W * H);
}

// img += amount * (img - blur(img)). scratchA, scratchB are each W*H bytes.
inline void unsharp(uint8_t *img, int W, int H, float amount, uint8_t *scratchA, uint8_t *scratchB) {
  memcpy(scratchA, img, (size_t)W * H);
  blur3(scratchA, W, H, scratchB);
  for (int i = 0; i < W * H; i++) {
    float v = img[i] + amount * (img[i] - scratchA[i]);
    img[i] = (uint8_t)iclamp((int)(v + 0.5f), 0, 255);
  }
}

// ---------------------------------------------------------------- tone / contrast
// Clip the darkest and brightest clipFraction of pixels and stretch the rest to 0..255.
inline void autoLevels(uint8_t *img, int n, float clipFraction, int *loOut, int *hiOut) {
  uint32_t hist[256];
  memset(hist, 0, sizeof(hist));
  for (int i = 0; i < n; i++) hist[img[i]]++;
  uint32_t clip = (uint32_t)(n * clipFraction);
  int lo = 0, hi = 255;
  uint32_t acc = 0;
  while (lo < 255 && acc + hist[lo] <= clip) acc += hist[lo++];
  acc = 0;
  while (hi > 0 && acc + hist[hi] <= clip) acc += hist[hi--];
  if (hi - lo < 16) {
    if (lo > 239) lo = 239;
    hi = lo + 16;
  }
  uint8_t lut[256];
  for (int i = 0; i < 256; i++) lut[i] = (uint8_t)iclamp((i - lo) * 255 / (hi - lo), 0, 255);
  for (int i = 0; i < n; i++) img[i] = lut[img[i]];
  if (loOut) *loOut = lo;
  if (hiOut) *hiOut = hi;
}

inline void equalize(uint8_t *img, int n) {
  uint32_t hist[256];
  memset(hist, 0, sizeof(hist));
  for (int i = 0; i < n; i++) hist[img[i]]++;
  int64_t cdf[256];
  int64_t run = 0, cdfMin = -1;
  for (int i = 0; i < 256; i++) {
    run += hist[i];
    cdf[i] = run;
    if (cdfMin < 0 && hist[i]) cdfMin = run;
  }
  uint8_t lut[256];
  for (int i = 0; i < 256; i++) {
    if (n == cdfMin) lut[i] = (uint8_t)i;
    else lut[i] = (uint8_t)iclamp((int)(((cdf[i] - cdfMin) * 255) / (n - cdfMin)), 0, 255);
  }
  for (int i = 0; i < n; i++) img[i] = lut[img[i]];
}

// Contrast-limited adaptive histogram equalization on a 4x4 grid of tiles.
// W and H must be divisible by 4.
inline void clahe(uint8_t *img, int W, int H, float clipFactor) {
  const int T = 4;
  static uint8_t lut[T * T][256];
  const int tw = W / T, th = H / T;
  const int area = tw * th;
  for (int ty = 0; ty < T; ty++)
    for (int tx = 0; tx < T; tx++) {
      uint32_t hist[256];
      memset(hist, 0, sizeof(hist));
      for (int y = ty * th; y < (ty + 1) * th; y++)
        for (int x = tx * tw; x < (tx + 1) * tw; x++) hist[img[y * W + x]]++;
      uint32_t limit = (uint32_t)imax(1, (int)(clipFactor * area / 256.0f));
      uint32_t excess = 0;
      for (int i = 0; i < 256; i++)
        if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; }
      uint32_t add = excess / 256, rem = excess % 256;
      for (int i = 0; i < 256; i++) hist[i] += add + ((uint32_t)i < rem ? 1 : 0);
      uint32_t run = 0;
      for (int i = 0; i < 256; i++) {
        run += hist[i];
        lut[ty * T + tx][i] = (uint8_t)imin(255, (int)((run * 255ULL) / (uint32_t)area));
      }
    }
  for (int y = 0; y < H; y++) {
    float fy = (y + 0.5f) / th - 0.5f;
    int y0 = (int)floorf(fy);
    float wy = fy - y0;
    int ya = iclamp(y0, 0, T - 1), yb = iclamp(y0 + 1, 0, T - 1);
    for (int x = 0; x < W; x++) {
      float fx = (x + 0.5f) / tw - 0.5f;
      int x0 = (int)floorf(fx);
      float wx = fx - x0;
      int xa = iclamp(x0, 0, T - 1), xb = iclamp(x0 + 1, 0, T - 1);
      uint8_t p = img[y * W + x];
      float top = lut[ya * T + xa][p] * (1 - wx) + lut[ya * T + xb][p] * wx;
      float bot = lut[yb * T + xa][p] * (1 - wx) + lut[yb * T + xb][p] * wx;
      img[y * W + x] = (uint8_t)iclamp((int)(top * (1 - wy) + bot * wy + 0.5f), 0, 255);
    }
  }
}

inline void applyGamma(uint8_t *img, int n, float gamma) {
  uint8_t lut[256];
  for (int i = 0; i < 256; i++) lut[i] = (uint8_t)iclamp((int)(255.0f * powf(i / 255.0f, gamma) + 0.5f), 0, 255);
  for (int i = 0; i < n; i++) img[i] = lut[img[i]];
}

// ---------------------------------------------------------------- dithering
// All dithers write a 1-bit image: bit = 1 means "burn a dot" (black).
static const uint8_t BAYER4[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
static const uint8_t BAYER8[8][8] = {
    {0, 32, 8, 40, 2, 34, 10, 42}, {48, 16, 56, 24, 50, 18, 58, 26},
    {12, 44, 4, 36, 14, 46, 6, 38}, {60, 28, 52, 20, 62, 30, 54, 22},
    {3, 35, 11, 43, 1, 33, 9, 41},  {51, 19, 59, 27, 49, 17, 57, 25},
    {15, 47, 7, 39, 13, 45, 5, 37}, {63, 31, 55, 23, 61, 29, 53, 21}};
// 45-degree clustered-dot (halftone) matrix, values 0..63
static const uint8_t CLUSTER8[8][8] = {
    {24, 10, 12, 26, 35, 47, 49, 37}, {8, 0, 2, 14, 45, 59, 61, 51},
    {22, 6, 4, 16, 43, 57, 63, 53},   {30, 20, 18, 28, 33, 41, 55, 39},
    {34, 46, 48, 36, 25, 11, 13, 27}, {44, 58, 60, 50, 9, 1, 3, 15},
    {42, 56, 62, 52, 23, 7, 5, 17},   {32, 40, 54, 38, 31, 21, 19, 29}};

struct Tap { int8_t dx, dy, w; };
static const Tap TAPS_FS[] = {{1, 0, 7}, {-1, 1, 3}, {0, 1, 5}, {1, 1, 1}};                       // /16
static const Tap TAPS_ATKINSON[] = {{1, 0, 1}, {2, 0, 1}, {-1, 1, 1}, {0, 1, 1}, {1, 1, 1}, {0, 2, 1}};  // /8
static const Tap TAPS_SIERRA_LITE[] = {{1, 0, 2}, {-1, 1, 1}, {0, 1, 1}};                         // /4
static const Tap TAPS_BURKES[] = {{1, 0, 8}, {2, 0, 4}, {-2, 1, 2}, {-1, 1, 4}, {0, 1, 8}, {1, 1, 4}, {2, 1, 2}};  // /32
static const Tap TAPS_STUCKI[] = {{1, 0, 8}, {2, 0, 4}, {-2, 1, 2}, {-1, 1, 4}, {0, 1, 8}, {1, 1, 4},
                                  {2, 1, 2}, {-2, 2, 1}, {-1, 2, 2}, {0, 2, 4}, {1, 2, 2}, {2, 2, 1}};         // /42
static const Tap TAPS_JJN[] = {{1, 0, 7}, {2, 0, 5}, {-2, 1, 3}, {-1, 1, 5}, {0, 1, 7}, {1, 1, 5},
                               {2, 1, 3}, {-2, 2, 1}, {-1, 2, 3}, {0, 2, 5}, {1, 2, 3}, {2, 2, 1}};            // /48

#define IMGPROC_MAXW 384

inline void errorDiffuse(const uint8_t *g, int W, int H, uint8_t *bits, int rowBytes,
                         const Tap *taps, int nTaps, int div, bool serpentine) {
  const int PAD = 2;
  static int16_t rows[3][IMGPROC_MAXW + 2 * PAD];  // ring of 3 error rows
  memset(rows, 0, sizeof(rows));
  memset(bits, 0, (size_t)rowBytes * H);
  for (int y = 0; y < H; y++) {
    int16_t *cur = rows[y % 3];
    bool rev = serpentine && (y & 1);
    for (int i = 0; i < W; i++) {
      int x = rev ? (W - 1 - i) : i;
      int v = g[y * W + x] + cur[x + PAD];
      int q = (v < 128) ? 0 : 255;
      int err = v - q;
      if (q == 0) bits[y * rowBytes + (x >> 3)] |= (uint8_t)(0x80 >> (x & 7));
      for (int t = 0; t < nTaps; t++) {
        int dx = rev ? -taps[t].dx : taps[t].dx;
        rows[(y + taps[t].dy) % 3][x + dx + PAD] += (int16_t)(err * taps[t].w / div);
      }
    }
    memset(cur, 0, sizeof(rows[0]));  // this row slot is reused for y + 3
  }
}

inline void dither(Dither d, const uint8_t *g, int W, int H, uint8_t *bits, int rowBytes) {
  switch (d) {
    case D_THRESH:
    case D_BAYER4:
    case D_BAYER8:
    case D_CLUSTER8:
    case D_BLUE:
      memset(bits, 0, (size_t)rowBytes * H);
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
          int thr;
          if (d == D_THRESH) thr = 128;
          else if (d == D_BAYER4) thr = BAYER4[y & 3][x & 3] * 16 + 8;
          else if (d == D_BAYER8) thr = BAYER8[y & 7][x & 7] * 4 + 2;
          else if (d == D_CLUSTER8) thr = CLUSTER8[y & 7][x & 7] * 4 + 2;
          else thr = BLUE64[(y & 63) * 64 + (x & 63)];
          if (g[y * W + x] < thr) bits[y * rowBytes + (x >> 3)] |= (uint8_t)(0x80 >> (x & 7));
        }
      break;
    case D_FS:           errorDiffuse(g, W, H, bits, rowBytes, TAPS_FS, 4, 16, false); break;
    case D_FS_SERP:      errorDiffuse(g, W, H, bits, rowBytes, TAPS_FS, 4, 16, true); break;
    case D_ATKINSON:     errorDiffuse(g, W, H, bits, rowBytes, TAPS_ATKINSON, 6, 8, false); break;
    case D_SIERRA_LITE:  errorDiffuse(g, W, H, bits, rowBytes, TAPS_SIERRA_LITE, 3, 4, false); break;
    case D_BURKES:       errorDiffuse(g, W, H, bits, rowBytes, TAPS_BURKES, 7, 32, false); break;
    case D_STUCKI:       errorDiffuse(g, W, H, bits, rowBytes, TAPS_STUCKI, 12, 42, false); break;
    case D_JJN:          errorDiffuse(g, W, H, bits, rowBytes, TAPS_JJN, 12, 48, false); break;
  }
}

// Replicate each dot into a factor x factor block (coarse-dot mode).
inline void expandBits(const uint8_t *small, int sW, int sH, int sRowBytes,
                       uint8_t *out, int outRowBytes, int factor) {
  memset(out, 0, (size_t)outRowBytes * sH * factor);
  for (int y = 0; y < sH; y++)
    for (int x = 0; x < sW; x++)
      if (small[y * sRowBytes + (x >> 3)] & (0x80 >> (x & 7)))
        for (int fy = 0; fy < factor; fy++)
          for (int fx = 0; fx < factor; fx++) {
            int ox = x * factor + fx, oy = y * factor + fy;
            out[oy * outRowBytes + (ox >> 3)] |= (uint8_t)(0x80 >> (ox & 7));
          }
}

// ---------------------------------------------------------------- full pipeline
struct Buffers {
  uint8_t *gray;       // outW*outH bytes
  uint8_t *scratchA;   // outW*outH bytes
  uint8_t *scratchB;   // outW*outH bytes
  uint8_t *bitsSmall;  // (outW/8)*outH bytes, used only for coarse > 1
};

// src: S x S grayscale. out: (outW/8) * outH bytes, 1 = black.
// Order: resize -> denoise -> sharpen -> contrast -> gamma -> dither.
inline void runPipeline(const Variant &v, const uint8_t *src, int S, int outW, int outH,
                        const Buffers &b, uint8_t *out, int *loOut, int *hiOut) {
  const int c = v.coarse < 1 ? 1 : v.coarse;
  const int W = outW / c, H = outH / c, n = W * H;
  resizeImage(src, S, b.gray, W, H, v.resize);
  if (v.blur == 1) blur3(b.gray, W, H, b.scratchA);
  else if (v.blur == 2) median3(b.gray, W, H, b.scratchA);
  if (v.usm > 0.0f) unsharp(b.gray, W, H, v.usm, b.scratchA, b.scratchB);
  if (v.eq) equalize(b.gray, n);
  else if (v.clahe) clahe(b.gray, W, H, 3.0f);
  int lo = -1, hi = -1;
  if (v.clip >= 0.0f) autoLevels(b.gray, n, v.clip, &lo, &hi);
  if (fabsf(v.gamma - 1.0f) > 0.001f) applyGamma(b.gray, n, v.gamma);
  if (c == 1) {
    dither(v.dither, b.gray, W, H, out, outW / 8);
  } else {
    dither(v.dither, b.gray, W, H, b.bitsSmall, W / 8);
    expandBits(b.bitsSmall, W, H, W / 8, out, outW / 8, c);
  }
  if (loOut) *loOut = lo;
  if (hiOut) *hiOut = hi;
}

}  // namespace imgproc