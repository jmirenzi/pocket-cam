// Thermal camera settings sweep: XIAO ESP32S3 Sense + 58mm TTL thermal printer (ESC/POS)
//
// Takes ONE photo, then prints it many times with different dithering / filtering / heat
// settings. Every strip is preceded by a text label listing exactly which settings produced it.
// The list of settings lives in sweep_table.h, the image maths in imgproc.h.
//
// USB serial monitor commands (115200):
//   s       capture one photo and print the whole sweep (one labeled strip per variant)
//   f <n>   then Enter: reprint variant n FULL SIZE from the same captured photo
//   c       capture and print one full-size photo with the BASE settings (also the D0 button)
//   g       grab a frame and report size + mean brightness (camera test, no printing)
//   t       print a text test (printer test, no camera)
//
// Wiring: ESP D6 (TX) -> printer RX, common GND, printer powered separately.

#include <Arduino.h>
#include "esp_camera.h"
#include "sweep_table.h"

using namespace imgproc;

// ---------------- Config ----------------
constexpr int PIN_BTN_CAPTURE = D0;  // button to GND, internal pull-up
constexpr int PIN_PRINTER_TX  = D6;  // ESP TX -> printer RX
constexpr int PIN_PRINTER_RX  = D7;  // ESP RX <- printer TX (leave unconnected for now)

constexpr uint32_t PRINTER_BAUD = 9600;  // see the printer self-test page for its baud rate

constexpr int PRINT_W   = 384;          // dots per line on a 58mm head
constexpr int PRINT_H   = 384;          // full-size square print
constexpr int ROW_BYTES = PRINT_W / 8;  // 48
constexpr int BAND_ROWS = 24;           // rows per GS v 0 command
constexpr int BAND_DELAY_MS = 40;       // let the printer catch up after each band

// Sweep strips print only part of the image so the paper roll lasts. Each strip is native
// scale (1 image pixel = 1 dot), so dither patterns look exactly like a full print.
// Set STRIP_H = PRINT_H (and STRIP_Y0 = 0) to print every variant full size.
constexpr int STRIP_H  = 160;
constexpr int STRIP_Y0 = (PRINT_H - STRIP_H) / 2;

// ESC 7: n1 = max dots heated at once (units of 8), n3 = heat interval. n2 (heat time) is per variant.
constexpr uint8_t HEAT_DOTS = 5, HEAT_INTERVAL = 2;
constexpr uint8_t DEFAULT_HEAT_TIME = 110;

constexpr bool CAM_VFLIP = false;
constexpr bool CAM_HMIRROR = false;

constexpr int SRC_SIDE = 600;   // SVGA is 800x600, we use the centered 600x600 square
constexpr int AVG_FRAMES = 4;   // frames averaged for the "4-FRAME AVERAGE" variant

// ---------------- Camera pins (XIAO ESP32S3 Sense, internal B2B connector) ----------------
#define XCLK_GPIO   10
#define SIOD_GPIO   40
#define SIOC_GPIO   39
#define VSYNC_GPIO  38
#define HREF_GPIO   47
#define PCLK_GPIO   13
#define Y2_GPIO     15
#define Y3_GPIO     17
#define Y4_GPIO     18
#define Y5_GPIO     16
#define Y6_GPIO     14
#define Y7_GPIO     12
#define Y8_GPIO     11
#define Y9_GPIO     48

HardwareSerial PrinterSerial(1);

static uint8_t *raw1 = nullptr;      // one captured frame, SRC_SIDE x SRC_SIDE
static uint8_t *rawAvg = nullptr;    // average of AVG_FRAMES captured frames
static uint16_t *accum = nullptr;    // accumulator for averaging
static uint8_t *bitmap = nullptr;    // 1-bit output, 1 = burn dot
static Buffers bufs;
static bool haveSource = false;

static void *allocBuf(size_t n) {
  void *p = ps_malloc(n);
  if (!p) p = malloc(n);
  return p;
}

// ---------------- Printer ----------------
void printerHeat(uint8_t heatTime) {
  const uint8_t cmd[] = {0x1B, 0x37, HEAT_DOTS, heatTime, HEAT_INTERVAL};  // ESC 7 n1 n2 n3
  PrinterSerial.write(cmd, sizeof(cmd));
}

void printerInit() {
  PrinterSerial.begin(PRINTER_BAUD, SERIAL_8N1, PIN_PRINTER_RX, PIN_PRINTER_TX);
  delay(200);
  const uint8_t init[] = {0x1B, 0x40};  // ESC @ : reset
  PrinterSerial.write(init, sizeof(init));
  printerHeat(DEFAULT_HEAT_TIME);
}

void printerFeed(uint8_t lines) {
  const uint8_t cmd[] = {0x1B, 0x64, lines};  // ESC d n
  PrinterSerial.write(cmd, sizeof(cmd));
}

void printerFont(uint8_t font) {  // 0 = font A (32 chars/line), 1 = font B (42 chars/line)
  const uint8_t cmd[] = {0x1B, 0x4D, font};  // ESC M n
  PrinterSerial.write(cmd, sizeof(cmd));
}

void printerText(const char *s) {
  PrinterSerial.print(s);
  PrinterSerial.print("\n");
}

void printerRaster(const uint8_t *bits, int rows) {
  for (int y = 0; y < rows; y += BAND_ROWS) {
    int h = (rows - y < BAND_ROWS) ? (rows - y) : BAND_ROWS;
    // GS v 0 m xL xH yL yH + data. xL/xH = bytes per row, yL/yH = rows
    const uint8_t hdr[] = {0x1D, 0x76, 0x30, 0x00,
                           (uint8_t)(ROW_BYTES & 0xFF), (uint8_t)(ROW_BYTES >> 8),
                           (uint8_t)(h & 0xFF), (uint8_t)(h >> 8)};
    PrinterSerial.write(hdr, sizeof(hdr));
    PrinterSerial.write(bits + y * ROW_BYTES, h * ROW_BYTES);
    PrinterSerial.flush();
    delay(BAND_DELAY_MS);
  }
}

void testPrintText() {
  Serial.println("Printing text test...");
  printerHeat(DEFAULT_HEAT_TIME);
  printerText("ESP32S3 thermal camera");
  printerText("Printer test OK");
  printerText("0123456789 ABCDEFGHIJ");
  printerFeed(3);
}

// ---------------- Camera ----------------
bool cameraInit() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO;
  cfg.pin_d1 = Y3_GPIO;
  cfg.pin_d2 = Y4_GPIO;
  cfg.pin_d3 = Y5_GPIO;
  cfg.pin_d4 = Y6_GPIO;
  cfg.pin_d5 = Y7_GPIO;
  cfg.pin_d6 = Y8_GPIO;
  cfg.pin_d7 = Y9_GPIO;
  cfg.pin_xclk = XCLK_GPIO;
  cfg.pin_pclk = PCLK_GPIO;
  cfg.pin_vsync = VSYNC_GPIO;
  cfg.pin_href = HREF_GPIO;
  cfg.pin_sccb_sda = SIOD_GPIO;
  cfg.pin_sccb_scl = SIOC_GPIO;
  cfg.pin_pwdn = -1;
  cfg.pin_reset = -1;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_GRAYSCALE;
  cfg.frame_size = FRAMESIZE_SVGA;  // 800x600
  cfg.jpeg_quality = 12;
  cfg.fb_count = 2;
  cfg.fb_location = CAMERA_FB_IN_PSRAM;
  cfg.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x (is PSRAM enabled? is the camera board seated?)\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, CAM_VFLIP);
    s->set_hmirror(s, CAM_HMIRROR);
  }
  for (int i = 0; i < 10; i++) {  // let auto-exposure settle
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(50);
  }
  return true;
}

// Grab one frame; copy its centered SRC_SIDE square to `copyTo` and/or add it into `acc`.
static bool grabInto(uint8_t *copyTo, uint16_t *acc) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return false;
  if ((int)fb->width < SRC_SIDE || (int)fb->height < SRC_SIDE) {
    Serial.printf("Unexpected frame size %dx%d\n", (int)fb->width, (int)fb->height);
    esp_camera_fb_return(fb);
    return false;
  }
  const int fbW = (int)fb->width, fbH = (int)fb->height;
  const int x0 = (fbW - SRC_SIDE) / 2, y0 = (fbH - SRC_SIDE) / 2;
  for (int y = 0; y < SRC_SIDE; y++) {
    const uint8_t *row = fb->buf + (y0 + y) * fbW + x0;
    if (copyTo) memcpy(copyTo + y * SRC_SIDE, row, SRC_SIDE);
    if (acc)
      for (int x = 0; x < SRC_SIDE; x++) acc[y * SRC_SIDE + x] += row[x];
  }
  esp_camera_fb_return(fb);
  return true;
}

// Capture the single frame (raw1) plus an AVG_FRAMES average (rawAvg) of the same scene.
bool captureSource() {
  Serial.println("Capturing...");
  for (int i = 0; i < 2; i++) {  // drop possibly stale frames
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  memset(accum, 0, (size_t)SRC_SIDE * SRC_SIDE * sizeof(uint16_t));
  if (!grabInto(raw1, accum)) {
    Serial.println("Frame grab failed");
    return false;
  }
  for (int i = 1; i < AVG_FRAMES; i++) {
    delay(60);
    if (!grabInto(nullptr, accum)) {
      Serial.println("Frame grab failed");
      return false;
    }
  }
  for (int i = 0; i < SRC_SIDE * SRC_SIDE; i++) rawAvg[i] = (uint8_t)(accum[i] / AVG_FRAMES);
  haveSource = true;
  Serial.println("Captured. Hold still: the 4-frame average needs the scene to stay put.");
  return true;
}

void testGrab() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame grab failed");
    return;
  }
  uint64_t sum = 0;
  for (size_t i = 0; i < fb->len; i += 16) sum += fb->buf[i];
  Serial.printf("Frame OK: %dx%d, %u bytes, mean brightness %u/255\n", (int)fb->width, (int)fb->height,
                (unsigned)fb->len, (unsigned)(sum / (fb->len / 16)));
  esp_camera_fb_return(fb);
}

// ---------------- Sweep ----------------
static void printLabel(const Variant &v, int idx, int total, bool full) {
  char line[80];
  char clip[12];
  if (v.clip < 0) snprintf(clip, sizeof clip, "off");
  else snprintf(clip, sizeof clip, "%.0f%%", v.clip * 100.0f);
  const char *blur = (v.blur == 0) ? "off" : (v.blur == 1 ? "gauss" : "median");

  printerFont(1);  // small font so each line fits in 42 characters
  if (total > 0) snprintf(line, sizeof line, "%s%02d/%02d %s", full ? "FULL " : "", idx + 1, total, v.name);
  else snprintf(line, sizeof line, "%s%s", full ? "FULL " : "", v.name);
  printerText(line);
  snprintf(line, sizeof line, "dither: %s", ditherName(v.dither));
  printerText(line);
  snprintf(line, sizeof line, "rs:%s blur:%s usm:%.1f clip:%s", resizeName(v.resize), blur, v.usm, clip);
  printerText(line);
  snprintf(line, sizeof line, "eq:%d clahe:%d g:%.1f avg4:%d heat:%d dot:%d", v.eq, v.clahe, v.gamma,
           v.avg4, v.heat, v.coarse);
  printerText(line);
  printerFont(0);
}

// Run the pipeline for one variant and print it (label first, then the image).
static void printVariant(const Variant &v, int idx, int total, bool full) {
  uint32_t t0 = millis();
  int lo, hi;
  runPipeline(v, v.avg4 ? rawAvg : raw1, SRC_SIDE, PRINT_W, PRINT_H, bufs, bitmap, &lo, &hi);
  Serial.printf("[%02d/%02d] %-22s processed in %lu ms (levels %d..%d)\n", idx + 1, total, v.name,
                (unsigned long)(millis() - t0), lo, hi);

  printerHeat(v.heat);
  printLabel(v, idx, total, full);
  const int rows = full ? PRINT_H : STRIP_H;
  const int y0 = full ? 0 : STRIP_Y0;
  printerRaster(bitmap + y0 * ROW_BYTES, rows);
  printerFeed(1);
}

static void runSweep() {
  if (!captureSource()) return;
  const int n = sweep::nVariants;
  Serial.printf("Printing %d strips...\n", n);

  char line[64];
  printerHeat(DEFAULT_HEAT_TIME);
  printerFont(1);
  snprintf(line, sizeof line, "SWEEP: %d strips, rows %d-%d", n, STRIP_Y0, STRIP_Y0 + STRIP_H - 1);
  printerText(line);
  snprintf(line, sizeof line, "base dither: %s", ditherName(sweep::BASE_DITHER));
  printerText(line);
  printerFont(0);
  printerFeed(1);

  for (int i = 0; i < n; i++) printVariant(sweep::variants[i], i, n, false);
  printerFeed(4);
  Serial.println("Sweep done. Use: f <n> then Enter, to reprint a favorite full size.");
}

static void printFinalist(int n) {
  if (!haveSource) {
    Serial.println("No photo yet. Run s first.");
    return;
  }
  if (n < 1 || n > sweep::nVariants) {
    Serial.printf("Pick a number from 1 to %d\n", sweep::nVariants);
    return;
  }
  printVariant(sweep::variants[n - 1], n - 1, sweep::nVariants, true);
  printerFeed(4);
}

static void captureAndPrintBase() {
  if (!captureSource()) return;
  Variant base = sweep::baseVariant();
  base.name = "BASE";
  printVariant(base, 0, 0, true);
  printerFeed(4);
  Serial.println("Done");
}

// ---------------- Arduino ----------------
static bool awaitingNumber = false;
static int numberAcc = 0, numberDigits = 0;

static void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (awaitingNumber) {
      if (c >= '0' && c <= '9') {
        numberAcc = numberAcc * 10 + (c - '0');
        numberDigits++;
      } else if (c == '\r' || c == '\n') {
        awaitingNumber = false;
        if (numberDigits > 0) printFinalist(numberAcc);
      }
      continue;
    }
    if (c == 's') runSweep();
    else if (c == 'c') captureAndPrintBase();
    else if (c == 'g') testGrab();
    else if (c == 't') testPrintText();
    else if (c == 'f') {
      awaitingNumber = true;
      numberAcc = 0;
      numberDigits = 0;
      Serial.println("Variant number, then Enter:");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);  // give USB CDC time to enumerate
  Serial.println("\nThermal camera sweep");

  pinMode(PIN_BTN_CAPTURE, INPUT_PULLUP);

  const size_t px = (size_t)PRINT_W * PRINT_H;
  raw1 = (uint8_t *)allocBuf((size_t)SRC_SIDE * SRC_SIDE);
  rawAvg = (uint8_t *)allocBuf((size_t)SRC_SIDE * SRC_SIDE);
  accum = (uint16_t *)allocBuf((size_t)SRC_SIDE * SRC_SIDE * sizeof(uint16_t));
  bitmap = (uint8_t *)allocBuf((size_t)ROW_BYTES * PRINT_H);
  bufs.gray = (uint8_t *)allocBuf(px);
  bufs.scratchA = (uint8_t *)allocBuf(px);
  bufs.scratchB = (uint8_t *)allocBuf(px);
  bufs.bitsSmall = (uint8_t *)allocBuf((size_t)ROW_BYTES * PRINT_H);
  if (!raw1 || !rawAvg || !accum || !bitmap || !bufs.gray || !bufs.scratchA || !bufs.scratchB || !bufs.bitsSmall) {
    Serial.println("Out of memory. Check that PSRAM is enabled in platformio.ini.");
    while (true) delay(1000);
  }

  sweep::buildSweep();
  printerInit();
  if (!cameraInit()) Serial.println("Camera NOT ready");
  else Serial.printf("Camera ready. %d sweep variants. Commands: s / f / c / g / t, or the D0 button.\n", sweep::nVariants);
}

void loop() {
  handleSerial();
  if (digitalRead(PIN_BTN_CAPTURE) == LOW) {
    delay(30);  // debounce
    if (digitalRead(PIN_BTN_CAPTURE) == LOW) {
      captureAndPrintBase();
      while (digitalRead(PIN_BTN_CAPTURE) == LOW) delay(10);
    }
  }
}