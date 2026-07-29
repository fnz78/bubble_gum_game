"""
bubble.py - Bubble animation: grow, shrink, pop, interpolation
"""

import math
import numpy as np
from utils import (
    draw_bubble, draw_pop_particles, create_pop_particles,
    update_pop_particles, lerp, lerp_color, rainbow_color,
    WHITE, PINK_LIGHT, PINK_MID, PINK_DEEP
)


class Bubble:
    """
    Represents a single player's bubble.

    size:         current display radius (smoothed)
    target_size:  what size we're interpolating toward
    max_size:     radius at which the bubble pops
    """

    MAX_SIZE = 130       # px radius – win condition
    MIN_SIZE = 0
    GROW_RATE = 1.8      # px per unit of blow intensity
    SHRINK_RATE = 0.35   # px per frame when not blowing
    LERP_SPEED = 0.15    # smoothing factor

    def __init__(self, player_id: int, max_size: int = None):
        self.player_id = player_id          # 0 or 1
        self.max_size = max_size or self.MAX_SIZE
        self.size = 0.0                     # smoothed display radius
        self.target_size = 0.0             # logical radius
        self.popped = False
        self.winner = False
        self.pop_particles = []
        self.pop_timer = 0.0               # counts frames after pop
        self.pop_duration = 60             # frames to show pop
        self._rainbow_t = 0.0
        self._pulse_t = 0.0
        # Bubble position on screen (set by game_logic based on layout)
        self.screen_pos = (0, 0)
        self.grow_rate = self.GROW_RATE
        self.shrink_rate = self.SHRINK_RATE

    def reset(self):
        self.size = 0.0
        self.target_size = 0.0
        self.popped = False
        self.winner = False
        self.pop_particles = []
        self.pop_timer = 0.0
        self._rainbow_t = 0.0
        self._pulse_t = 0.0

    def grow(self, intensity: float):
        """Grow bubble by blow intensity (clamped)."""
        if self.popped:
            return
        delta = intensity * self.grow_rate * 10
        self.target_size = min(self.target_size + delta, self.max_size)

    def decay(self):
        """Naturally shrink when not blowing."""
        if self.popped:
            return
        self.target_size = max(self.target_size - self.shrink_rate, 0.0)

    def update(self) -> bool:
        """
        Update bubble state each frame.
        Returns True if bubble just popped this frame.
        """
        self._rainbow_t = (self._rainbow_t + 0.008) % 1.0
        self._pulse_t   = (self._pulse_t   + 0.05)  % (2 * math.pi)

        if self.popped:
            self.pop_particles = update_pop_particles(self.pop_particles)
            self.pop_timer += 1
            return False

        # Smooth size interpolation
        self.size = lerp(self.size, self.target_size, self.LERP_SPEED)
        self.size = max(0.0, self.size)

        # Check pop condition
        if self.target_size >= self.max_size:
            self._trigger_pop()
            return True

        return False

    def _trigger_pop(self):
        self.popped = True
        self.size = self.max_size
        self.pop_particles = create_pop_particles(
            self.screen_pos, count=60, color=self._get_color()
        )

    def _get_color(self):
        """Color cycles through pink/rainbow based on size."""
        frac = self.size / max(self.max_size, 1)
        if frac < 0.5:
            return lerp_color(PINK_LIGHT, PINK_MID, frac * 2)
        else:
            return lerp_color(PINK_MID, PINK_DEEP, (frac - 0.5) * 2)

    def _get_display_radius(self) -> int:
        """Add subtle pulse animation to the display radius."""
        pulse = math.sin(self._pulse_t) * max(2, self.size * 0.02)
        return max(0, int(self.size + pulse))

    def draw(self, frame):
        """Draw the bubble (or pop particles if popped)."""
        cx, cy = int(self.screen_pos[0]), int(self.screen_pos[1])

        if self.popped:
            draw_pop_particles(frame, self.screen_pos, self.pop_particles)
            if self.winner and self.pop_timer < self.pop_duration // 2:
                # Flash "POP!" text zone – handled in main
                pass
            return

        r = self._get_display_radius()
        if r < 1:
            return

        color = self._get_color()
        draw_bubble(frame, (cx, cy), r, color)

    @property
    def fraction(self) -> float:
        """Return fill fraction 0-1."""
        return min(self.size / max(self.max_size, 1), 1.0)

    @property
    def is_alive(self) -> bool:
        return not self.popped

    @property
    def pop_finished(self) -> bool:
        return self.popped and self.pop_timer >= self.pop_duration
