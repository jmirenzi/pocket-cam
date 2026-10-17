"""live_view.py - take a photo with the pocket-cam, tune every setting, preview it as a thermal
print, and send it to the printer.

Press Space (or Capture) to grab a photo from the XIAO's OV2640 over USB serial (the firmware's
"r <n>" command). Every pipeline setting has its own control; changing one re-renders the same
photo straight away. Print sends exactly the 1-bit image shown, with the heat, density and a
label listing every setting.

  python sim/live_view.py                 # auto-detects the XIAO's COM port
  python sim/live_view.py --port COM4

Close the PlatformIO serial monitor first: only one program can hold the COM port.
Keys: Space capture, P print, S save PNGs to sim/out/.
"""
import argparse
import os
import struct
import sys
import threading
import time
from dataclasses import replace

try:
    import tkinter as tk
    from tkinter import ttk

    import numpy as np
    import serial
    from PIL import Image, ImageTk
    from serial.tools import list_ports

    import thermal_sim as ts
except ImportError as e:
    sys.exit(f"{e}\nThis Python ({sys.executable}) is missing packages. Either run with the Python "
             f"that has them, or install them into this one:\n"
             f'  "{sys.executable}" -m pip install numpy scipy pillow pyserial numba')

MAGIC = b"FRM1"
ESPRESSIF_VIDS = (0x303A, 0x2886)  # USB-Serial/JTAG, Seeed
AVG_FRAMES = 4  # same as the firmware's 4-FRAME AVERAGE


# ---------------------------------------------------------------- serial: frames in
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
            self.on_update(first, (acc / AVG_FRAMES + 0.5).astype(np.uint8), "photo ready")
        except serial.SerialException as e:
            self.on_update(None, None, f"serial error: {e}")


# ---------------------------------------------------------------- serial: print out
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

    def __init__(self, ser, bits, heat, density, label, on_status):
        super().__init__(daemon=True)
        self.ser, self.bits, self.heat, self.density = ser, bits, heat, density
        self.label, self.on_status = label, on_status

    def run(self):
        data = np.packbits(self.bits.astype(np.uint8), axis=1).tobytes()  # 48 bytes/row, MSB = left dot
        label = self.label.encode("ascii", "replace")[:255]
        density = 0xFF if self.density < 0 else self.density
        pkt = (b"PBM" + struct.pack("<HBBB", self.bits.shape[0], min(max(self.heat, 0), 255), density, len(label))
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


# ---------------------------------------------------------------- UI
DITHER_KEYS = list(ts.DITHERS)
CONTRASTS = ["none", "hist. equalize", "CLAHE"]
DENSITIES = ["printer default"] + [f"{ts.density_pct(d)}%  (DC2 # {d})" for d in range(32)]


class App:
    def __init__(self, root, ser, ss, out_dir):
        self.root, self.ser, self.ss, self.out_dir = root, ser, ss, out_dir
        self.blue = ts.load_blue_noise()
        self.presets = [ts.BASE] + ts.sweep_variants(extra=True)
        self.lock = threading.Lock()
        self.shot = {"one": None, "avg": None, "seq": 0}
        self.status = "press SPACE to take a photo"
        self.worker = None
        self.bits = None
        self.render_pending = False
        self.last_seq = -1
        self.images = {}  # keeps PhotoImage references alive

        root.title("pocket-cam print preview")
        self._build()
        self.load_preset(ts.BASE)
        for key, fn in (("<space>", self.capture), ("p", self.print_), ("P", self.print_), ("s", self.save),
                        ("S", self.save)):
            root.bind(key, lambda e, fn=fn: (fn(), "break")[1])
        root.after(100, self.poll)

    # ---- layout
    def _build(self):
        r = self.root
        side = ts.PRINT_W * self.ss
        left = ttk.Frame(r, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(r, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="n")

        self.v = {
            "dither": tk.StringVar(), "coarse": tk.StringVar(), "resize": tk.StringVar(), "blur": tk.StringVar(),
            "usm": tk.DoubleVar(), "contrast": tk.StringVar(), "levels": tk.BooleanVar(), "clip": tk.DoubleVar(),
            "gamma": tk.DoubleVar(), "avg4": tk.BooleanVar(), "heat": tk.IntVar(), "density": tk.StringVar(),
            "dot_gain": tk.DoubleVar(value=ts.PrintModel.dot_gain), "preset": tk.StringVar(),
            "view": tk.StringVar(value="print"),
        }

        row = 0

        def section(title):
            nonlocal row
            f = ttk.LabelFrame(left, text=title, padding=(8, 4))
            f.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            f.columnconfigure(1, weight=1)
            row += 1
            return f

        def combo(f, r_, label, var, values, width=22):
            ttk.Label(f, text=label).grid(row=r_, column=0, sticky="w", padx=(0, 6))
            c = ttk.Combobox(f, textvariable=var, values=values, state="readonly", width=width, takefocus=0)
            c.grid(row=r_, column=1, sticky="ew", pady=2)
            c.bind("<<ComboboxSelected>>", lambda e: self.root.focus_set())
            return c

        def scale(f, r_, label, var, lo, hi, res):
            ttk.Label(f, text=label).grid(row=r_, column=0, sticky="w", padx=(0, 6))
            s = tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", length="2.2i",
                         takefocus=0, highlightthickness=0)
            s.grid(row=r_, column=1, sticky="ew")
            s.bind("<ButtonRelease-1>", lambda e: self.root.focus_set())
            return s

        def check(f, r_, label, var):
            c = ttk.Checkbutton(f, text=label, variable=var, takefocus=0)
            c.grid(row=r_, column=0, columnspan=2, sticky="w", pady=2)
            return c

        f = section("Start from a sweep preset")
        combo(f, 0, "Preset", self.v["preset"], [p.name for p in self.presets]).bind(
            "<<ComboboxSelected>>", lambda e: (self.load_preset(self.preset_by_name(self.v["preset"].get())),
                                               self.root.focus_set()))

        f = section("Image processing (in pipeline order)")
        combo(f, 0, "Resize", self.v["resize"], ts.RESIZES)
        combo(f, 1, "Dot size", self.v["coarse"], ["1x1 (native)", "2x2", "3x3", "4x4"])
        combo(f, 2, "Denoise", self.v["blur"], ts.BLURS)
        scale(f, 3, "Unsharp", self.v["usm"], 0, 2, 0.1)
        combo(f, 4, "Contrast", self.v["contrast"], CONTRASTS)
        check(f, 5, "Auto-levels (stretch to full range)", self.v["levels"])
        scale(f, 6, "Levels clip %", self.v["clip"], 0, 10, 0.5)
        scale(f, 7, "Gamma", self.v["gamma"], 0.2, 2.0, 0.05)
        combo(f, 8, "Dither", self.v["dither"], [ts.DITHERS[k] for k in DITHER_KEYS])
        check(f, 9, "4-frame average (less sensor noise)", self.v["avg4"])

        f = section("Printer")
        scale(f, 0, "Heat time", self.v["heat"], 0, 255, 1)
        combo(f, 1, "Density", self.v["density"], DENSITIES)
        ttk.Label(f, text="Heat is one byte: 255 is the max. Density (DC2 #)\ngoes darker if your printer supports it.",
                  foreground="#666").grid(row=2, column=0, columnspan=2, sticky="w")

        f = section("Simulation only (not sent to the printer)")
        scale(f, 0, "Dot gain", self.v["dot_gain"], 0.10, 0.80, 0.01)

        f = section("Label that will print")
        self.label_text = tk.Label(f, justify="left", anchor="w", font=("Consolas", 9), width=44)
        self.label_text.grid(row=0, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, sticky="ew", pady=(4, 4))
        row += 1
        for i, (t, fn) in enumerate((("Capture [Space]", self.capture), ("Print [P]", self.print_),
                                     ("Save [S]", self.save))):
            ttk.Button(btns, text=t, command=fn, takefocus=0).grid(row=0, column=i, padx=(0, 6))
        self.status_lbl = ttk.Label(left, text="", wraplength="3.6i", foreground="#0a4")
        self.status_lbl.grid(row=row, column=0, sticky="w")

        tabs = ttk.Frame(view)
        tabs.grid(row=0, column=0, sticky="w")
        for i, (t, val) in enumerate((("Simulated print", "print"), ("1-bit data sent to printer", "bits"),
                                      ("Camera", "camera"))):
            ttk.Radiobutton(tabs, text=t, value=val, variable=self.v["view"], takefocus=0).grid(row=0, column=i, padx=(0, 12))
        # Start with a blank image: without one, a Label's width/height are in text characters,
        # and a 1152-character-wide label crashes Tk.
        self.images["main"] = ImageTk.PhotoImage(Image.new("RGB", (side, side), (85, 85, 85)))
        self.canvas = tk.Label(view, image=self.images["main"], bg="#555")
        self.canvas.grid(row=1, column=0, pady=(4, 0))

        for name, var in self.v.items():
            if name != "preset":
                var.trace_add("write", lambda *_: self.schedule_render())

    # ---- settings <-> Variant
    def preset_by_name(self, name):
        return next((p for p in self.presets if p.name == name), ts.BASE)

    def load_preset(self, p):
        self.loading = True
        v = self.v
        v["preset"].set(p.name)
        v["dither"].set(ts.DITHERS[p.dither])
        v["coarse"].set(["1x1 (native)", "2x2", "3x3", "4x4"][p.coarse - 1])
        v["resize"].set(p.resize)
        v["blur"].set(ts.BLURS[p.blur])
        v["usm"].set(p.usm)
        v["contrast"].set(ts.contrast_name(p))
        v["levels"].set(p.clip >= 0)
        v["clip"].set(max(p.clip, 0) * 100)
        v["gamma"].set(p.gamma)
        v["avg4"].set(p.avg4)
        v["heat"].set(p.heat)
        v["density"].set(DENSITIES[p.density + 1])
        self.loading = False
        self.schedule_render()

    def variant(self):
        v = self.v
        c = v["contrast"].get()
        var = ts.Variant(
            name=v["preset"].get(),
            dither=DITHER_KEYS[[ts.DITHERS[k] for k in DITHER_KEYS].index(v["dither"].get())],
            resize=v["resize"].get(), blur=ts.BLURS.index(v["blur"].get()), usm=round(v["usm"].get(), 2),
            clip=round(v["clip"].get(), 2) / 100 if v["levels"].get() else -1.0,
            eq=c == "hist. equalize", clahe=c == "CLAHE", gamma=round(v["gamma"].get(), 3),
            avg4=v["avg4"].get(), heat=int(v["heat"].get()), coarse=int(v["coarse"].get()[0]),
            density=DENSITIES.index(v["density"].get()) - 1)
        if var != self.preset_by_name(var.name):
            var = replace(var, name="CUSTOM")
        return var

    # ---- rendering
    def schedule_render(self):
        if not getattr(self, "loading", False) and not self.render_pending:
            self.render_pending = True
            self.root.after(30, self.render)

    def render(self):
        self.render_pending = False
        v = self.variant()
        self.label_text.config(text="\n".join(ts.label_lines(v)))
        with self.lock:
            one, avg = self.shot["one"], self.shot["avg"]
        src = avg if (v.avg4 and avg is not None) else one
        if src is None:
            return
        side = ts.PRINT_W * self.ss
        _, self.bits = ts.run_pipeline(v, src, self.blue)
        view = self.v["view"].get()
        if view == "print":
            m = ts.PrintModel(ss=self.ss, dot_gain=self.v["dot_gain"].get())
            img = Image.fromarray(ts.simulate_print(self.bits, v.heat, m, 1, v.density))
        elif view == "bits":
            img = Image.fromarray(((1 - self.bits) * 255).astype(np.uint8)).resize((side, side), Image.NEAREST)
        else:
            img = Image.fromarray(src).resize((side, side), Image.LANCZOS)
        self.images["main"] = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.images["main"])

    # ---- serial work
    def on_update(self, one, avg, status):
        with self.lock:
            if one is not None:
                self.shot["one"], self.shot["avg"] = one, avg
                self.shot["seq"] += 1
            self.status = status

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def capture(self):
        if self.busy():
            return
        self.on_update(None, None, "capturing...")
        self.worker = Capture(self.ser, self.on_update)
        self.worker.start()

    def print_(self):
        if self.busy():
            return
        if self.bits is None:
            self.on_update(None, None, "take a photo first")
            return
        v = self.variant()
        self.on_update(None, None, "sending to printer...")
        self.worker = Print(self.ser, self.bits, v.heat, v.density, "\n".join(ts.label_lines(v)),
                            lambda s: self.on_update(None, None, s))
        self.worker.start()

    def save(self):
        with self.lock:
            one, avg = self.shot["one"], self.shot["avg"]
        v = self.variant()
        src = avg if (v.avg4 and avg is not None) else one
        if src is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        _, bits = ts.run_pipeline(v, src, self.blue)
        m = ts.PrintModel(ss=4, dot_gain=self.v["dot_gain"].get())
        page = ts.on_paper(ts.simulate_print(bits, v.heat, m, 1, v.density), m, ts.label_lines(v), 1)
        paths = [os.path.join(self.out_dir, f"live_{stamp}_{s}.png") for s in ("print", "bits", "camera")]
        page.save(paths[0])
        Image.fromarray(((1 - bits) * 255).astype(np.uint8)).save(paths[1])
        Image.fromarray(src).save(paths[2])
        self.on_update(None, None, f"saved {os.path.basename(paths[0])} (+ bits, camera)")

    def poll(self):
        with self.lock:
            seq, avg_ready, status = self.shot["seq"], self.shot["avg"] is not None, self.status
        key = (seq, avg_ready)
        if key != self.last_seq:
            self.last_seq = key
            self.schedule_render()
        v_avg4 = self.v["avg4"].get()
        note = "  (4-frame average not ready yet, showing one frame)" if v_avg4 and not avg_ready and seq else ""
        self.status_lbl.config(text=status + note)
        self.root.after(100, self.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM4 (default: auto-detect)")
    ap.add_argument("--ss", type=int, help="screen pixels per printer dot (default: fit the screen, max 3)")
    ap.add_argument("--out", default=os.path.join(ts.HERE, "out"))
    args = ap.parse_args()

    if sys.platform == "win32":  # crisp text and true pixel sizes on scaled displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass

    port = args.port or find_port()
    print(f"opening {port}")
    ser = open_port(port)
    root = tk.Tk()
    # biggest preview that fits under the tabs and taskbar; 3+ px per dot shows the dot shapes
    ss = args.ss or max(1, min(3, (root.winfo_screenheight() - 300) // ts.PRINT_H))
    App(root, ser, ss, args.out)
    root.mainloop()
    ser.close()


if __name__ == "__main__":
    main()
