"""
This is a simple experiment to see how to use Manim to create visualizations. This file
in particular creates a top-down view of a tiled grid of squares where the color of each square is
determined by an altitude value that is computed from a set of layered noise functions.

It can be rendered with the following command:

```
uv run manim -pql terrain_scene.py TerrainScene
```

The `-pql` flag is used to render the video in low quality. It means:

- `p`: render a preview
- `ql`: use the lowest quality setting
"""

from manim import *
from bisect import bisect_right
import numpy as np

def color_from_altitude(altitude, color_ramp_params):
    index = bisect_right([t for t,_ in color_ramp_params], altitude)
    if index == 0:
        return color_ramp_params[0][1]
    if index == len(color_ramp_params):
        return color_ramp_params[-1][1]
    t1, c1 = color_ramp_params[index - 1]
    t2, c2 = color_ramp_params[index]
    return interpolate_color(c1, c2, (altitude - t1) / (t2 - t1))

class TerrainScene(Scene):
    def construct(self):
        GRID_RESOLUTION = 128
        GRID_SIDE_LENGTH = 1 / GRID_RESOLUTION
        GRID_PADDING_PCT = 0.15
        PADDED_GRID_SIDE_LENGTH = GRID_SIDE_LENGTH * (1 + GRID_PADDING_PCT)

        color_ramp_params = [
            (0.0, DARK_BLUE),
            (0.43, DARK_BLUE),
            (0.55, BLUE),
            (0.58, WHITE),
            (0.60, LIGHT_BROWN),
            (0.62, GREEN),
            (0.70, GREEN),
            (0.75, GREY),
            (1.0, WHITE),
        ]

        noise_params = [
            {
                "frequency": 2,
                "amplitude": 1,
            },
            {
                "frequency": 4,
                "amplitude": 0.5,
            },
            {
                "frequency": 8,
                "amplitude": 0.25,
            },
            {
                "frequency": 16,
                "amplitude": 0.125,
            },
            {
                "frequency": 32,
                "amplitude": 0.0625,
            },
        ]

        # Standard permutation list for Perlin noise
        permutation = np.array([
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
            69,142,8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,
            252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,
            171,168, 68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,
            229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
            65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
            250,124,123, 5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,
            16,58,17,182,189,28,42,223,183,170,213,119,248,152, 2,44,154,163,
            70,221,153,101,155,167, 43,172,9,129,22,39,253, 19,98,108,110,79,
            113,224,232,178,185, 112,104,218,246,97,228,251,34,242,193,238,
            210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,
            214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,
            254,138,236,205,93,222,114, 67,29,24,72,243,141,128,195,78,66,
            215,61,156,180
        ], dtype=int)
        permutation = np.concatenate([permutation, permutation])  # Extend the table

        def perlin_noise(coords, frequency=4.0, amplitude=1.0):
            """
            Calculates the Perlin noise value at each point in coords.
            coords has shape (2, H, W) so coords[0][i, j] is the x coordinate,
            and coords[1][i, j] is the y coordinate.
            """

            def fade(t):
                return 6*t**5 - 15*t**4 + 10*t**3

            def grad(hash_val, x, y):
                # Simple gradient selection based on the lowest 2 bits of hash
                g = hash_val & 3
                if g == 0:
                    return  x + y
                elif g == 1:
                    return  x - y
                elif g == 2:
                    return -x + y
                else:
                    return -x - y

            # coords: (2, H, W)
            H, W = coords.shape[1], coords.shape[2]
            noise_output = np.zeros((H, W))

            # Scale coords by frequency
            x_coords = coords[0] * frequency
            y_coords = coords[1] * frequency

            for i in range(H):
                for j in range(W):
                    x = x_coords[i, j]
                    y = y_coords[i, j]

                    # Find integer "grid cell" corners
                    X = int(np.floor(x)) & 255
                    Y = int(np.floor(y)) & 255

                    # Fractional parts
                    xf = x - np.floor(x)
                    yf = y - np.floor(y)

                    # Hash coordinates of the corners
                    top_right = permutation[permutation[X + 1] + Y + 1]
                    top_left  = permutation[permutation[X    ] + Y + 1]
                    bottom_right = permutation[permutation[X + 1] + Y]
                    bottom_left  = permutation[permutation[X    ] + Y]

                    # Fade curves
                    u = fade(xf)
                    v = fade(yf)

                    # Interpolate
                    n0 = grad(bottom_left, xf, yf)
                    n1 = grad(bottom_right, xf - 1, yf)
                    ix1 = np.interp(u, [0,1], [n0, n1])

                    n2 = grad(top_left, xf, yf - 1)
                    n3 = grad(top_right, xf - 1, yf - 1)
                    ix2 = np.interp(u, [0,1], [n2, n3])

                    value = np.interp(v, [0,1], [ix1, ix2])
                    noise_output[i, j] = value

            return amplitude * noise_output

        def calculate_normalized_altitude(coords, noise_params):
            """
            Calculates the normalized altitude using layered noise functions.
            """
            alts = sum(perlin_noise(coords, p["frequency"], p["amplitude"]) for p in noise_params)
            min_alt = -sum(p["amplitude"] for p in noise_params)
            max_alt = sum(p["amplitude"] for p in noise_params)
            normalized = (alts - min_alt) / (max_alt - min_alt)
            return normalized
        
        # coords[x, y] = (x, y)
        coords = np.stack(
            [
                np.arange(0, 1, GRID_SIDE_LENGTH)[:, np.newaxis].repeat(GRID_RESOLUTION, axis=1),
                np.arange(0, 1, GRID_SIDE_LENGTH)[np.newaxis, :].repeat(GRID_RESOLUTION, axis=0),
            ]
        )
        coords = coords + np.array([0.5, 0.5])[:, np.newaxis, np.newaxis]
        normalized_altitudes = calculate_normalized_altitude(coords, noise_params)

        # Create squares colored by altitude
        squares = VGroup()

        for i in range(GRID_RESOLUTION):
            for j in range(GRID_RESOLUTION):
                alt = normalized_altitudes[i, j]
                color = color_from_altitude(alt, color_ramp_params)

                sq = Square(side_length=GRID_SIDE_LENGTH, fill_opacity=1, stroke_width=0)
                sq.set_fill(color)

                # Position each square so that the entire grid is near the center
                sq.move_to(
                    [
                        (i - (GRID_RESOLUTION / 2 + 0.5)) * PADDED_GRID_SIDE_LENGTH,
                        (j - (GRID_RESOLUTION / 2 + 0.5)) * PADDED_GRID_SIDE_LENGTH,
                        0,
                    ]
                )
                squares.add(sq)

        # Optionally scale the entire group for visibility
        squares.scale(5)

        self.play(Create(squares))

        self.wait(5)
