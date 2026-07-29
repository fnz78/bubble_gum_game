"""
utils.py - Drawing helpers, color utilities, and text rendering
"""

import cv2
import numpy as np
import math


# ── Color Palette ──────────────────────────────────────────────────────────────
PINK_LIGHT  = (210, 170, 255)   # BGR
PINK_MID    = (180, 100, 255)
PINK_DEEP   = (120,  50, 220)
BLUE_GLOW   = (255, 200,  80)
WHITE       = (255, 255, 255)
BLACK       = (0,   0,   0)
GREEN_NEON  = (100, 255, 100)
RED_NEON    = (80,  80,  255)
YELLOW_NEON = (50,  230, 255)
CYAN_NEON   = (255, 230,  60)

P1_COLOR = (255, 140,  80)   # Player 1: orange-ish
P2_COLOR = (80,  190, 255)   # Player 2: blue-ish


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def draw_text_shadow(frame, text, pos, font=cv2.FONT_HERSHEY_DUPLEX,
                     scale=1.0, color=WHITE, thickness=2, shadow_offset=2):
    """Draw text with a drop shadow."""
    shadow_c = (0, 0, 0)
    sx, sy = pos[0] + shadow_offset, pos[1] + shadow_offset
    cv2.putText(frame, text, (sx, sy), font, scale, shadow_c, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(frame, text, cy, font=cv2.FONT_HERSHEY_DUPLEX,
                        scale=1.0, color=WHITE, thickness=2):
    """Draw horizontally centered text."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    draw_text_shadow(frame, text, (x, cy), font, scale, color, thickness)


def draw_glow_circle(frame, center, radius, color, alpha=0.5, layers=4):
    """Draw a glowing circle with multiple semi-transparent layers."""
    overlay = frame.copy()
    for i in range(layers, 0, -1):
        r = int(radius + i * 6)
        a = alpha * (i / layers) * 0.5
        c_layer = tuple(min(255, int(c + (255 - c) * 0.1)) for c in color)
        cv2.circle(overlay, center, r, c_layer, -1)
        cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)
        overlay = frame.copy()


def draw_bubble(frame, center, radius, color, highlight=True, pop_frac=0.0):
    """
    Draw a bubble with gradient, specular highlights, and transparency.
    pop_frac: 0.0 = normal, 1.0 = fully popped
    """
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r < 3:
        return

    # Glow rings
    glow_layers = 5
    for i in range(glow_layers, 0, -1):
        gr = r + i * 4
        ga = 0.06
        gcolor = tuple(min(255, c + 40) for c in color)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), gr, gcolor, 2)
        cv2.addWeighted(overlay, ga, frame, 1 - ga, 0, frame)

    # Main bubble body (semi-transparent fill)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Bubble outline
    cv2.circle(frame, (cx, cy), r, color, 2, cv2.LINE_AA)

    # Inner ring
    if r > 15:
        cv2.circle(frame, (cx, cy), int(r * 0.85), color, 1, cv2.LINE_AA)

    # Specular highlight (top-left)
    if highlight and r > 10:
        h_cx = cx - int(r * 0.3)
        h_cy = cy - int(r * 0.35)
        h_r  = max(3, int(r * 0.22))
        ov2  = frame.copy()
        cv2.circle(ov2, (h_cx, h_cy), h_r, WHITE, -1)
        cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)

        # Small secondary highlight
        h2_r = max(2, int(r * 0.1))
        ov3  = frame.copy()
        cv2.circle(ov3, (h_cx + int(r * 0.18), h_cy + int(r * 0.05)),
                   h2_r, WHITE, -1)
        cv2.addWeighted(ov3, 0.5, frame, 0.5, 0, frame)


def draw_pop_particles(frame, center, particles):
    """
    Draw explosion particles from a popped bubble.
    Each particle: dict with x, y, vx, vy, life (0-1), color, size
    """
    for p in particles:
        if p["life"] <= 0:
            continue
        alpha = p["life"]
        r = max(1, int(p["size"] * alpha))
        cx, cy = int(p["x"]), int(p["y"])
        if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
            col = tuple(int(c * alpha) for c in p["color"])
            cv2.circle(frame, (cx, cy), r, col, -1, cv2.LINE_AA)


def draw_progress_bar(frame, pos, size, value, max_value, color, label=""):
    """Draw a horizontal progress bar."""
    x, y = pos
    w, h = size
    filled = int(w * min(value / max(max_value, 1), 1.0))

    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    # Fill
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)

    # Label
    if label:
        draw_text_shadow(frame, label, (x, y - 6),
                         scale=0.45, color=WHITE, thickness=1)


def create_pop_particles(center, count=40, color=(210, 170, 255)):
    """Generate particles for pop animation."""
    particles = []
    cx, cy = center
    for _ in range(count):
        angle = np.random.uniform(0, 2 * math.pi)
        speed = np.random.uniform(3, 12)
        lifespan = np.random.uniform(0.5, 1.0)
        size = np.random.randint(3, 10)
        jitter_c = tuple(min(255, max(0, c + np.random.randint(-40, 40)))
                         for c in color)
        particles.append({
            "x": float(cx),
            "y": float(cy),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": lifespan,
            "max_life": lifespan,
            "color": jitter_c,
            "size": size,
        })
    return particles


def update_pop_particles(particles, dt=1.0):
    """Update particle positions and lifetimes."""
    alive = []
    for p in particles:
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        p["vy"] += 0.3 * dt   # gravity
        p["vx"] *= 0.97        # drag
        p["life"] -= dt * 0.04
        if p["life"] > 0:
            # Re-normalize to 0-1
            p["life_frac"] = p["life"] / p["max_life"]
            alive.append(p)
    return alive


def draw_dark_overlay(frame, alpha=0.55):
    """Darken the entire frame."""
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def hsv_to_bgr(h, s, v):
    """Convert HSV (0-179, 0-255, 0-255) to BGR."""
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])


def rainbow_color(t):
    """Return a cycling rainbow BGR color based on t (0-1)."""
    h = int(t * 179) % 180
    return hsv_to_bgr(h, 230, 255)
