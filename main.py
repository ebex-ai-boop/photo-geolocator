#!/usr/bin/env python3
"""photo-geolocator-01: model-free street-photo geolocation.

Approach (deterministic, no model calls, no network except the env API):
  1. Decode each JPEG with a pure-stdlib decoder (baseline/extended
     sequential + progressive DC, 4:4:4/4:2:2/4:2:0) to full YCbCr planes.
  2. Measure a handful of coarse visual features on half-resolution
     planes, plus a strict yellow-paint detector on full-resolution
     pixels in the road band.
  3. Apply a small decision list of general world-geography rules
     (validated by leave-one-trial-out cross-validation on the 3
     practice trials; see README notes in the repo):
       - strict yellow centre line  -> the Americas (yellow paint is
         the standard centre-line colour there), split by climate
         into US Southwest / Brazil / central North America;
       - warm, arid, bright ground -> the subtropical desert belt
         (Arabian Peninsula / Middle East);
       - dense green, dim           -> the wet tropics (SE Asia);
       - otherwise                  -> the spherical geometric median
         of the practice pool, (56.4, 30.66), the best single prior.
  4. POST all 25 answers at once; fall back to per-photo POSTs.
     A photo that fails to download or decode still gets the prior
     guess (never left unanswered: unanswered = 20,015 km).

Practice-trial performance (honest, leave-one-trial-out):
  mean total error 160,346 km per 25-photo trial (mean 6,414 km/photo,
  score ~340,031 of 500,377) vs 171,168 km for the prior alone.
  Within 2,500 km: 18/75 photos. Errors stay large: this is a weak
  signal, not a precise locator.
"""
import json
import math
import os
import struct
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------- JPEG ---
class JpegError(Exception):
    pass


class BitReader:
    """MSB-first bit reader over the entropy segment, with 0xFF00 unstuffing
    and restart-marker skipping. Bits past the end of data read as 1."""

    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.buf = 0
        self.nbits = 0
        # End-of-scan padding: once the EOI marker is reached, the
        # encoder's final partial byte is padded with 1-bits, so further
        # reads return 0xFF. Guarded: more than 64 padded bits means the
        # decoder has really desynced, not just finished the last MCU.
        self.eof = False
        self.padded = 0

    def _next_byte(self):
        d = self.data
        p = self.pos
        if self.eof:
            self.padded += 8
            if self.padded > 64:
                raise JpegError("entropy data exhausted")
            return 0xFF
        if p >= len(d):
            self.eof = True
            return self._next_byte()
        b = d[p]
        if b != 0xFF:
            self.pos = p + 1
            return b
        if p + 1 >= len(d):
            self.eof = True
            return self._next_byte()
        nb = d[p + 1]
        if nb == 0x00:
            self.pos = p + 2
            return 0xFF
        if 0xD0 <= nb <= 0xD7:
            # restart marker: byte-aligned; reset bit buffer
            self.pos = p + 2
            self.buf = 0
            self.nbits = 0
            return self._next_byte()
        if nb == 0xD9:
            # EOI: the remaining bits of the last byte are 1-padding.
            self.eof = True
            return self._next_byte()
        raise JpegError("marker 0xFF%02x inside scan" % nb)

    def bits(self, n):
        while self.nbits < n:
            self.buf = (self.buf << 8) | self._next_byte()
            self.nbits += 8
        self.nbits -= n
        v = (self.buf >> self.nbits) & ((1 << n) - 1)
        self.buf &= (1 << self.nbits) - 1 if self.nbits else 0
        return v

    def align(self):
        self.nbits = 0
        self.buf = 0


class HuffTable:
    def __init__(self, counts, symbols):
        self._peek = {}
        self._walk = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                sym = symbols[k]
                k += 1
                self._walk[(length, code)] = sym
                if length <= 8:
                    fill = 8 - length
                    base = code << fill
                    for rest in range(1 << fill):
                        self._peek[base | rest] = (sym, length)
                code += 1
            code <<= 1

    def decode(self, br):
        while br.nbits < 8:
            br.buf = (br.buf << 8) | br._next_byte()
            br.nbits += 8
        top = (br.buf >> (br.nbits - 8)) & 0xFF
        hit = self._peek.get(top)
        if hit is not None:
            sym, length = hit
            br.nbits -= length
            br.buf &= (1 << br.nbits) - 1 if br.nbits else 0
            return sym
        code = 0
        for length in range(1, 17):
            code = (code << 1) | br.bits(1)
            sym = self._walk.get((length, code))
            if sym is not None:
                return sym
        raise JpegError("bad Huffman code")


def _extend(v, size):
    if size == 0:
        return 0
    if v < (1 << (size - 1)):
        return v - (1 << size) + 1
    return v


# 1-D IDCT matrix T[k][i] = 0.5 * C(k) * cos((2i+1)*k*pi/16)
_T = []
for _k in range(8):
    _c = 1.0 / math.sqrt(2.0) if _k == 0 else 1.0
    _T.append([0.5 * _c * math.cos((2 * _i + 1) * _k * math.pi / 16.0)
               for _i in range(8)])
_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7 = _T
_ZIG = []
for _s in range(15):
    _pts = [(_r, _s - _r) for _r in range(8) if 0 <= _s - _r < 8]
    if _s % 2 == 0:
        _pts.reverse()
    _ZIG.extend(_pts)
_SQ = 0.3535533905932738  # one 1-D IDCT pass of a DC-only row/column


def _idct_block(coef):
    """coef: 64 dequantised coefficients in zigzag order -> 64 samples."""
    b = [0.0] * 64
    nz = 0
    for idx, (r, c) in enumerate(_ZIG):
        v = coef[idx]
        if v:
            b[r * 8 + c] = float(v)
            nz += 1
    if nz <= 1:
        return [b[0] * 0.125] * 64
    tmp = [0.0] * 64
    for r in range(8):
        base = r * 8
        b0, b1, b2, b3, b4, b5, b6, b7 = b[base:base + 8]
        if b1 == 0 and b2 == 0 and b3 == 0 and b4 == 0 and b5 == 0 and b6 == 0 and b7 == 0:
            v = b0 * _SQ
            for i in range(8):
                tmp[base + i] = v
            continue
        t0, t1, t2, t3, t4, t5, t6, t7 = _T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7
        for i in range(8):
            tmp[base + i] = (t0[i] * b0 + t1[i] * b1 + t2[i] * b2 + t3[i] * b3
                             + t4[i] * b4 + t5[i] * b5 + t6[i] * b6 + t7[i] * b7)
    out = [0.0] * 64
    for c in range(8):
        c0, c1, c2, c3, c4, c5, c6, c7 = (tmp[c], tmp[8 + c], tmp[16 + c], tmp[24 + c],
                                         tmp[32 + c], tmp[40 + c], tmp[48 + c], tmp[56 + c])
        if c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0 and c5 == 0 and c6 == 0 and c7 == 0:
            v = c0 * _SQ
            for r in range(8):
                out[r * 8 + c] = v
            continue
        t0, t1, t2, t3, t4, t5, t6, t7 = _T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7
        for r in range(8):
            out[r * 8 + c] = (t0[r] * c0 + t1[r] * c1 + t2[r] * c2 + t3[r] * c3
                              + t4[r] * c4 + t5[r] * c5 + t6[r] * c6 + t7[r] * c7)
    return out


def decode_planes(data):
    """JPEG -> (W, H, Y, Cb, Cr): flat row-major int lists, Cb/Cr
    nearest-neighbour upsampled to full resolution."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise JpegError("not a JPEG (no SOI)")
    pos = 2
    qtab = {}
    huff_dc = {}
    huff_ac = {}
    width = height = 0
    comps = []
    restart_interval = 0
    scans = []
    while True:
        if pos + 1 >= len(data) or data[pos] != 0xFF:
            raise JpegError("marker expected at %d" % pos)
        marker = data[pos + 1]
        pos += 2
        if marker == 0xD9:
            break
        if marker == 0xDA:
            n = struct.unpack(">H", data[pos:pos + 2])[0]
            ns = data[pos + 2]
            scomps = []
            for i in range(ns):
                cid = data[pos + 3 + 2 * i]
                td = data[pos + 4 + 2 * i]
                scomps.append((cid, td >> 4, td & 15))
            ss = data[pos + n - 3]
            start = pos + n
            scans.append((start, scomps, ss))
            i = start
            nd = len(data)
            while i + 1 < nd:
                if data[i] == 0xFF:
                    nb = data[i + 1]
                    if nb == 0x00 or 0xD0 <= nb <= 0xD7:
                        i += 2
                        continue
                    break
                i += 1
            pos = i
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        n = struct.unpack(">H", data[pos:pos + 2])[0]
        seg = data[pos + 2:pos + n]
        if marker == 0xDB:
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tq = seg[j] & 15
                size = 64 * (2 if pq else 1)
                vals = seg[j + 1:j + 1 + size]
                if pq:
                    vals = [struct.unpack(">H", vals[2 * k:2 * k + 2])[0]
                            for k in range(64)]
                else:
                    vals = list(vals)
                qtab[tq] = vals
                j += 1 + size
        elif marker in (0xC0, 0xC1, 0xC2):
            height = struct.unpack(">H", seg[1:3])[0]
            width = struct.unpack(">H", seg[3:5])[0]
            nc = seg[5]
            comps = []
            for i in range(nc):
                cid = seg[6 + 3 * i]
                hv = seg[7 + 3 * i]
                tq = seg[8 + 3 * i]
                comps.append((cid, hv >> 4, hv & 15, tq))
        elif marker == 0xC4:
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                th = seg[j] & 15
                counts = list(seg[j + 1:j + 17])
                total = sum(counts)
                symbols = list(seg[j + 17:j + 17 + total])
                tbl = HuffTable(counts, symbols)
                if tc == 0:
                    huff_dc[th] = tbl
                else:
                    huff_ac[th] = tbl
                j += 17 + total
        elif marker == 0xDD:
            restart_interval = struct.unpack(">H", seg[:2])[0]
        pos += n
    if not scans or not comps:
        raise JpegError("no scan found")
    if width <= 0 or height <= 0:
        raise JpegError("no SOF found")
    cmap = {c[0]: c for c in comps}
    max_h = max(c[1] for c in comps)
    max_v = max(c[2] for c in comps)
    mcu_w = (width + 8 * max_h - 1) // (8 * max_h)
    mcu_h = (height + 8 * max_v - 1) // (8 * max_v)
    blocks = {}
    for cid, h, v, tq in comps:
        blocks[cid] = [[None] * (mcu_w * h) for _ in range(mcu_h * v)]
    done = set()
    for start, scomps, ss in scans:
        if ss != 0:
            continue  # progressive AC scans are not needed for our features
        br = BitReader(data, start)
        pred = {cid: 0 for cid, _, _, _ in comps}
        order = []
        for cid, td, ta in scomps:
            c = cmap[cid]
            order.append((cid, c[1], c[2], c[3], td, ta))
        for mcu in range(mcu_w * mcu_h):
            if restart_interval and mcu and mcu % restart_interval == 0:
                br.align()
                for cid in pred:
                    pred[cid] = 0
            my, mx = divmod(mcu, mcu_w)
            for cid, h, v, tq, td, ta in order:
                dct = huff_dc[td]
                act = huff_ac[ta]
                qt = qtab[tq]
                grid = blocks[cid]
                for by in range(v):
                    for bx in range(h):
                        coef = [0] * 64
                        s = dct.decode(br)
                        if s:
                            pred[cid] += _extend(br.bits(s), s)
                        coef[0] = pred[cid] * qt[0]
                        k = 1
                        while k < 64:
                            sym = act.decode(br)
                            if sym == 0x00:
                                break
                            run = sym >> 4
                            size = sym & 15
                            if size:
                                val = _extend(br.bits(size), size)
                            else:
                                val = 0
                            k += run
                            if k < 64 and size:
                                coef[k] = val * qt[k]
                            k += 1
                        grid[my * v + by][mx * h + bx] = coef
        for cid, _, _, _, _, _ in order:
            done.add(cid)
        if len(done) >= len(comps):
            break
    if len(done) < len(comps):
        raise JpegError("incomplete data")
    planes = {}
    for cid, h, v, tq in comps:
        bw = mcu_w * h
        bh = mcu_h * v
        tw = (width * h + max_h - 1) // max_h
        th = (height * v + max_v - 1) // max_v
        plane = [0] * (tw * th)
        grid = blocks[cid]
        for by in range(bh):
            for bx in range(bw):
                out = _idct_block(grid[by][bx])
                for r in range(8):
                    gy = by * 8 + r
                    if gy >= th:
                        break
                    dst = gy * tw + bx * 8
                    src = r * 8
                    for c in range(8):
                        gx = bx * 8 + c
                        if gx >= tw:
                            break
                        val = out[src + c] + 128.0
                        plane[dst + c] = 0 if val < 0 else (255 if val > 255 else int(val + 0.5))
        planes[cid] = (tw, th, plane)
    if len(comps) == 1:
        return width, height, planes[comps[0][0]][2], planes[comps[0][0]][2], planes[comps[0][0]][2]
    yid, cbid, crid = comps[0][0], comps[1][0], comps[2][0]
    Y = planes[yid][2]
    Cb = _stretch(planes[cbid], width, height)
    Cr = _stretch(planes[crid], width, height)
    return width, height, Y, Cb, Cr


def _stretch(pl, W, H):
    tw, th, p = pl
    if tw == W and th == H:
        return p
    out = [0] * (W * H)
    for y in range(H):
        sy = y * th // H
        src = sy * tw
        dst = y * W
        for x in range(W):
            out[dst + x] = p[src + x * tw // W]
    return out


# -------------------------------------------------------------- features ---
def _half(p, W, H):
    w2, h2 = W // 2, H // 2
    out = [0] * (w2 * h2)
    for y in range(h2):
        s0 = (2 * y) * W
        s1 = (2 * y + 1) * W
        d = y * w2
        for x in range(w2):
            out[d + x] = (p[s0 + 2 * x] + p[s0 + 2 * x + 1]
                          + p[s1 + 2 * x] + p[s1 + 2 * x + 1] + 2) // 4
    return out, w2, h2


def measure(W, H, Y, Cb, Cr):
    """Coarse visual features. Returns a dict; 'prior' on any failure."""
    # strict yellow paint in the road band (full-res, sampled every 2 px):
    # yellow paint has very low Cb and high Cr (Cb<100, Cr>140); red dirt
    # and sand do not reach such low Cb. Fraction of road-band pixels.
    c = n = 0
    for y in range(H // 2, H, 2):
        base = y * W
        for x in range(W // 5, 4 * W // 5, 2):
            i = base + x
            if Cb[i] < 100 and Cr[i] > 140:
                c += 1
            n += 1
    strict = c / n if n else 0.0

    # half-resolution planes for the remaining (climate) features
    Yh, w, h = _half(Y, W, H)
    Cbh, _, _ = _half(Cb, W, H)
    Crh, _, _ = _half(Cr, W, H)

    def is_sky(y, cb, cr):
        return cb > 140 and y > 90

    def frac(pred, y0=0, y1=None, x0=0, x1=None):
        y1 = h if y1 is None else y1
        x1 = w if x1 is None else x1
        cnt = tot = 0
        for yy in range(y0, y1):
            base = yy * w
            for xx in range(x0, x1):
                i = base + xx
                if pred(Yh[i], Cbh[i], Crh[i]):
                    cnt += 1
                tot += 1
        return cnt / tot if tot else 0.0

    sky_top = frac(is_sky, 0, h // 6)
    sky_bot = frac(is_sky, 5 * h // 6, h)
    if sky_bot > sky_top + 0.2 and sky_bot > 0.25:
        # photo is upside down (sky at the bottom): flip vertically
        Yh = [Yh[(h - 1 - yy) * w + xx] for yy in range(h) for xx in range(w)]
        Cbh = [Cbh[(h - 1 - yy) * w + xx] for yy in range(h) for xx in range(w)]
        Crh = [Crh[(h - 1 - yy) * w + xx] for yy in range(h) for xx in range(w)]

    def is_veg(y, cb, cr):
        return cb < 122 and cr < 118 and 40 < y < 185

    def is_arid(y, cb, cr):
        return cr > 138 and cb < 128 and y > 115

    def warm_frac(thr):
        cnt = tot = 0
        for yy in range(h // 2, h, 2):
            base = yy * w
            for xx in range(w // 5, 4 * w // 5, 2):
                i = base + xx
                if Crh[i] - Cbh[i] > thr:
                    cnt += 1
                tot += 1
        return cnt / tot if tot else 0.0

    N = w * h
    return {
        "strict": strict,
        "warm18": warm_frac(18),
        "arid_low": frac(is_arid, h // 3, h),
        "veg_low": frac(is_veg, h // 2, h),
        "bright": sum(Yh) / N if N else 128.0,
    }


# ------------------------------------------------------------------ rules ---
# Best single prior: spherical geometric median of the practice pool.
PRIOR = (56.4, 30.66)


def guess(f):
    """Decision list; f is the feature dict from measure()."""
    if f["strict"] > 0.008:
        # yellow centre line -> the Americas
        if f["warm18"] > 0.25:
            return (30.0, -105.0)   # US Southwest / northern Mexico
        if f["veg_low"] > 0.03:
            return (-12.0, -55.0)   # Brazil
        return (25.0, -95.0)        # central North America
    if f["warm18"] > 0.35 and f["arid_low"] > 0.04:
        return (26.0, 44.0)         # subtropical desert belt (Middle East)
    if f["veg_low"] > 0.05 and f["bright"] < 130:
        return (2.0, 100.0)         # wet tropics (SE Asia)
    return PRIOR


# -------------------------------------------------------------------- env ---
def _get(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _post(url, payload, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    env = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env:
        print("MAKEFUN_ENV_URL is not set", file=sys.stderr)
        return 2
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as fh:
                fh.read()
        except OSError:
            pass
    try:
        listing = json.loads(_get(env + "/photos").decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("failed to list photos: %r" % (exc,), file=sys.stderr)
        return 2
    photos = sorted(listing.get("photos", []), key=lambda p: p["id"])
    answers = []
    for p in photos:
        pid = p["id"]
        lat, lon = PRIOR
        try:
            data = None
            last = None
            for _ in range(3):
                try:
                    data = _get(env + "/photos/" + pid + ".jpg")
                    break
                except (urllib.error.URLError, OSError) as exc:
                    last = exc
            if data is None:
                raise last if last else JpegError("empty download")
            W, H, Y, Cb, Cr = decode_planes(data)
            f = measure(W, H, Y, Cb, Cr)
            lat, lon = guess(f)
            print("%s strict=%.4f warm18=%.2f arid_low=%.2f veg_low=%.3f "
                  "bright=%.0f -> %.4f,%.4f" % (pid, f["strict"], f["warm18"],
                  f["arid_low"], f["veg_low"], f["bright"], lat, lon))
        except Exception as exc:  # decode/feature failure: still answer
            print("%s FAILED (%r); using prior" % (pid, exc), file=sys.stderr)
            lat, lon = PRIOR
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))
        answers.append({"id": pid, "lat": round(lat, 6), "lon": round(lon, 6)})
    if not answers:
        print("no photos listed", file=sys.stderr)
        return 2
    payload = {"answers": answers}
    try:
        res = _post(env + "/answers", payload)
        print("POST all: %r" % (res,))
        if len(res.get("accepted", [])) == len(answers):
            return 0
    except Exception as exc:
        print("bulk POST failed: %r; falling back to per-photo POSTs" % (exc,),
              file=sys.stderr)
    ok = 0
    for a in answers:
        try:
            res = _post(env + "/answers", {"answers": [a]})
            if a["id"] in res.get("accepted", []):
                ok += 1
        except Exception as exc:
            print("POST %s failed: %r" % (a["id"], exc), file=sys.stderr)
    print("accepted %d of %d" % (ok, len(answers)))
    return 0 if ok == len(answers) else 1


if __name__ == "__main__":
    sys.exit(main())
