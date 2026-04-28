"""Locate the brightest local maxima in StarMap.png.

Outputs JSON to stdout: {"width": W, "height": H, "stars": [{"x": nx, "y": ny, "v": brightness}, ...]}
Coordinates are normalized to [0, 1] in image space.
"""

import json

import numpy as np
from PIL import Image, ImageFilter

PATH = "/home/jcena/STL/Andre-STL/StarMap.png"
TOP_N = 28
NEIGHBORHOOD = 18
THRESHOLD = 170


def main():
    img = Image.open(PATH).convert("L")
    w, h = img.size
    arr = np.asarray(img, dtype=np.uint8)

    dilated = np.asarray(img.filter(ImageFilter.MaxFilter(size=2 * (NEIGHBORHOOD // 2) + 1)), dtype=np.uint8)
    peak_mask = (arr == dilated) & (arr >= THRESHOLD)
    ys, xs = np.where(peak_mask)
    vals = arr[ys, xs]

    order = np.argsort(-vals)
    chosen = []
    min_dist_sq = (NEIGHBORHOOD * 1.5) ** 2
    for idx in order:
        x, y, v = int(xs[idx]), int(ys[idx]), int(vals[idx])
        ok = True
        for cx, cy, _ in chosen:
            if (cx - x) ** 2 + (cy - y) ** 2 < min_dist_sq:
                ok = False
                break
        if ok:
            chosen.append((x, y, v))
        if len(chosen) >= TOP_N:
            break

    stars = [
        {"x": round(x / w, 4), "y": round(y / h, 4), "v": v}
        for x, y, v in chosen
    ]
    print(json.dumps({"width": w, "height": h, "stars": stars}, indent=2))


if __name__ == "__main__":
    main()
