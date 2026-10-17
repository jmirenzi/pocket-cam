"""live_view.py - preview a real pocket-cam photo as a thermal print, then tune it live.

Press Space to grab a photo from the XIAO's OV2640 over USB serial (the firmware's "r <n>"
command). The photo runs through the same pipeline and paper simulation as thermal_sim.py and is
shown as camera | simulated print. Switching variants or moving the sliders re-renders the
same photo instantly, so you can compare settings on one shot.

  python sim/live_view.py                 # auto-detects the XIAO's COM port
  python sim/live_view.py --port COM4 --variant 7

Close the PlatformIO serial monitor first: only one program can hold the COM port.

Keys:  space  capture a new photo       ]  next variant     [  previous variant
       p      print the right-hand image on the real printer (exact same bits, heat and label)
       b      left panel: camera / 1-bit data sent to the printer
       s      save PNGs to sim/out/     q / Esc  quit
Sliders override the current variant's gamma, heat and unsharp, and the paper's dot gain.
"""
import argparse
import os
import struct
import sys
import threading
import time
from dataclasses import replace

try:
    import cv2
    import numpy as np
    import serial
    from serial.tools import list_ports

    import thermal_sim as ts
except ImportError as e:
    sys.exit(f"{e}\nThis Python ({sys.executable}) is missing packages. Either run with the Python "
             f"that has them, or install them into this one:\n"
             f'  "{sys.executable}" -m pip install numpy scipy pillow opencv-python pyserial numba')

MAGIC = b"FRM1"
ESPRESSIF_VIDS = (0x303A, 0x2886)  # USB-Serial/JTAG, Seeed
WIN = "pocket-cam live"
AVG_FRAMES = 4  # same as the firmware's 4-FRAME AVERAGE


# ---------------------------------------------------------------- serial frames
def find_port():
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in ESPRESSIF_VIDS:
            return p.device
    names = ", ".join(p.device for p in ports) or "none"
    sys.exit(f"No XIAO ESP32S3 found (ports: {names}). Pass --port COMx.")


def open_port(name):
    # serial_for_url also accepts socket://host:port, handy for testing without the board
    ser = serial.serial_for_url(name, do_not_open=True, timeout=0.5)
    ser.dtr = False  # DTR/RTS toggles can reset the ESP32-S3 through its USB-Serial/JTAG
    ser.rts = False
    try:
        ser.open()
    except serial.SerialException as e:
        sys.exit(f"Could not open {name}: {e}\nIs the PlatformIO serial monitor, an upload, or another "
                 f"live_view window using it? Close that and try again.")
    if hasattr(ser, "set_buffer_size"):  # Windows: let the driver buffer a whole frame
        ser.set_buffer_size(rx_size=1 << 20)
    return ser


def request_frame(ser, n=ts.SRC_SIDE, timeout=3.0):
    """Ask the firmware for one n x n gray frame. Returns a uint8 array or None.
    Gives up as soon as the data stalls for one read timeout (0.5 s), so grab() can retry."""
    ser.reset_input_buffer()
    ser.write(f"r{n}\n".encode())
    junk = ser.read_until(MAGIC, 65536)
    text = junk[:-len(MAGIC)] if junk.endswith(MAGIC) else junk
    if text.strip() and all(32 <= b < 127 or b in (9, 10, 13) for b in text):
        print(text.decode().rstrip())  # firmware log lines (button presses, errors); binary leftovers are skipped
    if not junk.endswith(MAGIC):
        return None
    hdr = ser.read(4)
    if len(hdr) < 4:
        return None
    w, h = struct.unpack("<HH", hdr)
    if not (0 < w <= 1024 and 0 < h <= 1024):
        return None
    need, chunks, deadline = w * h, [], time.time() + timeout
    while need > 0 and time.time() < deadline:
        c = ser.read(need)
        if not c:
            break
        chunks.append(c)
        need -= len(c)
    if need > 0:
        return None
    return np.frombuffer(b"".join(chunks), np.uint8).reshape(h, w)


def grab(ser, tries=3):
    for _ in range(tries):
        f = request_frame(ser)
        if f is not None:
            return f
    return None


class Capture(threading.Thread):
    """One photo plus, afterwards, the extra frames for the 4-frame average (like captureSource)."""

    def __init__(self, ser, on_update):
        super().__init__(daemon=True)
        self.ser, self.on_update = ser, on_update

    def run(self):
        try:
            first = grab(self.ser)
            if first is None:
                self.on_update(None, None, "capture failed (firmware flashed? serial monitor closed?)")
                return
            self.on_update(first, None, f"averaging 1/{AVG_FRAMES}...")
            acc = first.astype(np.float32)
            for i in range(1, AVG_FRAMES):
                f = grab(self.ser)
                if f is None:
                    self.on_update(first, None, "4-frame average failed")
                    return
                acc += f
                if i < AVG_FRAMES - 1:
                    self.on_update(first, None, f"averaging {i + 1}/{AVG_FRAMES}...")
            self.on_update(first, (acc / AVG_FRAMES + 0.5).astype(np.uint8), "")
        except serial.SerialException as e:
            self.on_update(None, None, f"serial error: {e}")


def read_line(ser, timeout):
    """One text line from the firmware, or None after `timeout` seconds of silence."""
    buf, deadline = b"", time.time() + timeout
    while time.time() < deadline:
        buf += ser.read_until(b"\n")
        if buf.endswith(b"\n"):
            return buf.decode(errors="replace").strip()
    return None


class Print(threading.Thread):
    """Send a finished 1-bit image to the firmware's PBM command and wait for the printer."""

    def __init__(self, ser, bits, heat, label, on_status):
        super().__init__(daemon=True)
        self.ser, self.bits, self.heat, self.label, self.on_status = ser, bits, heat, label, on_status

    def run(self):
        data = np.packbits(self.bits.astype(np.uint8), axis=1).tobytes()  # 48 bytes/row, MSB = left dot
        label = self.label.encode("ascii", "replace")[:255]
        pkt = (b"PBM" + struct.pack("<HBB", self.bits.shape[0], min(max(self.heat, 1), 255), len(label))
               + label + data + struct.pack("<H", sum(data) & 0xFFFF))
        try:
            self.ser.reset_input_buffer()
            self.ser.write(pkt)
            reply = read_line(self.ser, 3)
            if not reply or not reply.startswith("PRINT OK"):
                self.on_status(f"print failed: {reply or 'no reply (firmware flashed?)'}")
                return
            self.on_status("printing...")
            reply = read_line(self.ser, 90)  # 9600 baud to the printer: a full photo takes ~20 s
            self.on_status("printed" if reply == "PRINT DONE" else f"print: {reply or 'no PRINT DONE reply'}")
        except serial.SerialException as e:
            self.on_status(f"serial error: {e}")


# ---------------------------------------------------------------- rendering
def render(frame, v, m, blue, show_bits=False, seed=1):
    """Returns (bgr panel image, bits). frame: square gray camera image."""
    _, bits = ts.run_pipeline(v, frame, blue)
    sim = ts.simulate_print(bits, v.heat, m, seed)
    side = sim.shape[0]
    left = (1 - bits) * 255 if show_bits else frame
    interp = cv2.INTER_NEAREST if show_bits or left.shape[0] < side else cv2.INTER_AREA
    left = cv2.resize(left.astype(np.uint8), (side, side), interpolation=interp)
    panel = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sim, cv2.COLOR_RGB2BGR)])
    return panel, bits


def status_bar(width, lines):
    bar = np.full((22 * len(lines) + 10, width, 3), 32, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return bar


def placeholder(side, text):
    img = np.full((side, side * 2, 3), 60, np.uint8)
    cv2.putText(img, text, (20, side // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- UI
SLIDERS = [  # name, scale, min, max, variant field (None = print model)
    ("gamma x100", 100, 20, 200, "gamma"),
    ("heat", 1, 40, 250, "heat"),
    ("unsharp x10", 10, 0, 20, "usm"),
    ("dot gain x100", 100, 10, 80, None),
]


def make_window(m):
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, scale, lo, hi, _ in SLIDERS:
        cv2.createTrackbar(name, WIN, lo, hi, lambda _x: None)
        cv2.setTrackbarMin(name, WIN, lo)
    cv2.setTrackbarPos("dot gain x100", WIN, int(round(m.dot_gain * 100)))


def load_sliders(v):
    for name, scale, lo, hi, field in SLIDERS:
        if field:
            cv2.setTrackbarPos(name, WIN, int(np.clip(round(getattr(v, field) * scale), lo, hi)))


def read_sliders(v, m):
    for name, scale, _lo, _hi, field in SLIDERS:
        val = cv2.getTrackbarPos(name, WIN) / scale
        if field == "heat":
            v = replace(v, heat=int(val))
        elif field:
            v = replace(v, **{field: val})
        else:
            m = replace(m, dot_gain=max(val, 0.05))
    return v, m


def save(frame, v, m, blue, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _, bits = ts.run_pipeline(v, frame, blue)
    hi = replace(m, ss=4)
    page = ts.on_paper(ts.simulate_print(bits, v.heat, hi, seed), hi, ts.label_lines(v), seed)
    p1 = os.path.join(out_dir, f"live_{stamp}_print.png")
    p2 = os.path.join(out_dir, f"live_{stamp}_camera.png")
    page.save(p1)
    cv2.imwrite(p2, frame)
    print(f"saved {p1}\n      {p2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM4 (default: auto-detect)")
    ap.add_argument("--variant", "-v", type=int, default=5, help="starting variant (see thermal_sim.py --list)")
    ap.add_argument("--extra", action="store_true", help="include the SWEEP_EXTRA variants")
    ap.add_argument("--ss", type=int, default=2, help="screen pixels per printer dot")
    ap.add_argument("--out", default=os.path.join(ts.HERE, "out"))
    args = ap.parse_args()

    variants = ts.sweep_variants(args.extra)
    idx = min(max(args.variant, 1), len(variants)) - 1
    m = ts.PrintModel(ss=args.ss)
    blue = ts.load_blue_noise()

    port = args.port or find_port()
    print(f"opening {port}")
    ser = open_port(port)

    shot = {"one": None, "avg": None, "status": "press SPACE to take a photo", "seq": 0}
    lock = threading.Lock()

    def on_update(one, avg, status):
        with lock:
            if one is not None:
                shot["one"], shot["avg"] = one, avg
                shot["seq"] += 1
            shot["status"] = status

    make_window(m)
    load_sliders(variants[idx])
    show_bits = False
    worker, last_key, panel, bits = None, None, None, None  # worker: the one capture/print thread
    side = ts.PRINT_W * args.ss

    while True:
        v, mm = read_sliders(variants[idx], m)
        with lock:
            one, avg, status, seq = shot["one"], shot["avg"], shot["status"], shot["seq"]
        busy = worker is not None and worker.is_alive()
        src = avg if (v.avg4 and avg is not None) else one

        key = (seq, avg is not None, idx, v, mm, show_bits)
        if src is not None and key != last_key:
            last_key = key
            panel, bits = render(src, v, mm, blue, show_bits)

        note = "  (4-frame average not ready, showing single frame)" if v.avg4 and avg is None and one is not None else ""
        lines = [f"{idx + 1}/{len(variants)} {v.name}   dither {v.dither}   gamma {v.gamma:.2f}   heat {v.heat}"
                 f"   usm {v.usm:.1f}   dot gain {mm.dot_gain:.2f}",
                 f"{status}{note}   left: {'1-bit data' if show_bits else 'camera'}",
                 "SPACE capture   [ ] variant   p PRINT   b bits   s save   q quit"]
        body = panel if panel is not None else placeholder(side, status)
        cv2.imshow(WIN, np.vstack([body, status_bar(body.shape[1], lines)]))

        k = cv2.waitKey(15) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord(" ") and not busy:
            on_update(None, None, "capturing...")
            worker = Capture(ser, on_update)
            worker.start()
        elif k == ord("p") and bits is not None and not busy:
            on_update(None, None, "sending to printer...")
            worker = Print(ser, bits, v.heat, "\n".join(ts.label_lines(v)), lambda s: on_update(None, None, s))
            worker.start()
        elif k == ord("]"):
            idx = (idx + 1) % len(variants)
            load_sliders(variants[idx])
        elif k == ord("["):
            idx = (idx - 1) % len(variants)
            load_sliders(variants[idx])
        elif k == ord("b"):
            show_bits = not show_bits
        elif k == ord("s") and src is not None:
            save(src, v, mm, blue, args.out, 1)

    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
