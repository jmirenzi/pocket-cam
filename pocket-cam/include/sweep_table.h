// sweep_table.h - the list of settings combinations to print. Edit this to change the sweep.
// Pure C++11 so the same table can be exercised on a PC.
#pragma once
#include "imgproc.h"

namespace sweep {
using namespace imgproc;

// Dither used by every strip that is NOT testing a dither (preprocessing and heat strips).
// Once you know your favorite dither, change this and re-run.
static const Dither BASE_DITHER = D_FS;

// Set to 1 to append the extended list (more dithers, resize modes, denoise, clip, etc.).
#ifndef SWEEP_EXTRA
#define SWEEP_EXTRA 0
#endif

#define MAX_VARIANTS 40
static Variant variants[MAX_VARIANTS];
static int nVariants = 0;

static Variant baseVariant() {
  Variant v;
  v.name = "BASE";
  v.dither = BASE_DITHER;
  v.resize = R_BOX;
  v.blur = 0;
  v.usm = 0.0f;
  v.clip = 0.02f;
  v.eq = false;
  v.clahe = false;
  v.gamma = 1.0f;
  v.avg4 = false;
  v.heat = 110;
  v.coarse = 1;
  return v;
}

// Appends a variant that starts as BASE and returns it so the caller can change fields.
static Variant &add(const char *name) {
  variants[nVariants] = baseVariant();
  variants[nVariants].name = name;
  return variants[nVariants++];
}

static void buildSweep() {
  nVariants = 0;

  // ---- dithering algorithms (everything else = BASE) ----
  add("THRESHOLD").dither = D_THRESH;
  add("BAYER 8x8").dither = D_BAYER8;
  add("CLUSTERED DOT").dither = D_CLUSTER8;
  add("BLUE NOISE").dither = D_BLUE;
  add("FLOYD-STEINBERG").dither = D_FS;
  add("FS SERPENTINE").dither = D_FS_SERP;
  add("ATKINSON").dither = D_ATKINSON;
  add("SIERRA LITE").dither = D_SIERRA_LITE;

  // ---- preprocessing (BASE_DITHER) ----
  add("UNSHARP 0.5").usm = 0.5f;
  add("UNSHARP 1.0").usm = 1.0f;
  add("GAMMA 0.8 (LIGHTER)").gamma = 0.8f;
  {
    Variant &v = add("CLAHE LOCAL CONTRAST");
    v.clahe = true;
    v.clip = -1.0f;  // CLAHE already spreads the range, skip auto-levels
  }
  add("4-FRAME AVERAGE").avg4 = true;

  // ---- printer heat (BASE_DITHER) ----
  add("HEAT 80").heat = 80;
  add("HEAT 140").heat = 140;

#if SWEEP_EXTRA
  add("BAYER 4x4").dither = D_BAYER4;
  add("BURKES").dither = D_BURKES;
  add("STUCKI").dither = D_STUCKI;
  add("JARVIS-JUDICE-NINKE").dither = D_JJN;
  add("RESIZE NEAREST").resize = R_NEAREST;
  add("RESIZE BILINEAR").resize = R_BILINEAR;
  add("MEDIAN 3x3 DENOISE").blur = 2;
  add("BLUR 3x3").blur = 1;
  add("LEVELS CLIP 0%").clip = 0.0f;
  add("LEVELS CLIP 5%").clip = 0.05f;
  {
    Variant &v = add("HIST EQUALIZE");
    v.eq = true;
    v.clip = -1.0f;
  }
  add("COARSE DOTS 2x2").coarse = 2;
#endif
}

}  // namespace sweep