"""
game_logic.py - Game states, countdown, player management, win detection
"""

import time
import numpy as np
from enum import Enum, auto
from bubble import Bubble
from utils import P1_COLOR, P2_COLOR


class GameState(Enum):
    MENU        = auto()
    ASSIGN      = auto()   # face assignment for 2-player
    COUNTDOWN   = auto()
    PLAYING     = auto()
    PAUSED      = auto()
    WINNER      = auto()
    RESET       = auto()


class BlowDetector:
    """
    Detects blowing by monitoring the DECREASE in lip openness
    (closing mouth after opening = exhale push).
    """
    HISTORY_LEN = 6
    BLOW_THRESHOLD = 0.018   # minimum drop to count as blow
    NOISE_GATE    = 0.004    # ignore tiny fluctuations

    def __init__(self):
        self._history = []
        self._prev = None

    def update(self, openness: float) -> float:
        """
        Feed current openness ratio. Returns blow intensity (0-1).
        A positive value means the player is actively blowing.
        """
        self._history.append(openness)
        if len(self._history) > self.HISTORY_LEN:
            self._history.pop(0)

        if len(self._history) < 2:
            return 0.0

        # Blow = decrease in openness (mouth closes after opening)
        delta = self._history[-2] - self._history[-1]

        if delta < self.NOISE_GATE:
            return 0.0

        intensity = max(0.0, (delta - self.NOISE_GATE) /
                        (self.BLOW_THRESHOLD * 5 + 1e-6))
        return min(intensity, 1.0)

    def reset(self):
        self._history.clear()
        self._prev = None


class Player:
    """Represents one player (human face)."""

    def __init__(self, player_id: int, color, max_bubble_size=130):
        self.player_id = player_id
        self.color = color
        self.bubble = Bubble(player_id, max_size=max_bubble_size)
        self.blow_detector = BlowDetector()
        self.face_landmarks = None       # latest landmarks from tracker
        self.face_center_x = None       # for left/right assignment
        self.openness = 0.0
        self.blow_intensity = 0.0
        self.is_active = False
        self.mouth_center = None

    def process_face(self, landmarks, metrics: dict):
        """Feed new face data into the player."""
        self.face_landmarks = landmarks
        self.openness = metrics["openness_ratio"]
        self.blow_intensity = self.blow_detector.update(self.openness)
        self.is_active = True
        if "center_pt" in metrics:
            self.mouth_center = tuple(metrics["center_pt"])

    def update(self):
        """Called every frame."""
        # Bubble grows if active and blowing, decays otherwise (stops frozen bubble bug)
        if self.is_active and self.blow_intensity > 0:
            self.bubble.grow(self.blow_intensity)
        else:
            self.bubble.decay()

    def reset(self):
        self.face_landmarks = None
        self.face_center_x = None
        self.openness = 0.0
        self.blow_intensity = 0.0
        self.is_active = False
        self.mouth_center = None
        self.bubble.reset()
        self.blow_detector.reset()


class GameLogic:
    """
    Central game controller.

    Manages:
    - Game states (menu -> countdown -> playing -> paused -> winner)
    - Player list (1 or 2 players)
    - Face-to-player assignment
    - Countdown timer
    - Winner detection
    """

    COUNTDOWN_SECS = 3
    WIN_HOLD_SECS  = 4      # how long to show winner screen

    def __init__(self, num_players: int = 1, max_bubble_size: int = 130):
        self.num_players = num_players
        self.max_bubble_size = max_bubble_size
        self.state = GameState.MENU
        self._countdown_start = None
        self._winner_start = None
        self.winner_id = None    # 0 or 1
        
        # Mouse interactions
        self.mouse_x = -1
        self.mouse_y = -1
        self.pending_action = None
        
        # Screen dimensions
        self.screen_w = 1280
        self.screen_h = 720
        self.difficulty = "MEDIUM"

        colors = [P1_COLOR, P2_COLOR]
        self.players = [
            Player(i, colors[i], max_bubble_size)
            for i in range(num_players)
        ]
        self.apply_difficulty()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_game(self):
        """Begin the countdown."""
        if self.state in (GameState.MENU, GameState.WINNER, GameState.ASSIGN):
            self._reset_players()
            self.state = GameState.COUNTDOWN
            self._countdown_start = time.time()

    def reset(self):
        """Full reset to menu."""
        self._reset_players()
        self.state = GameState.MENU
        self._countdown_start = None
        self._winner_start = None
        self.winner_id = None
        self.pending_action = None
        self.apply_difficulty()

    def toggle_pause(self):
        """Toggle game pause state."""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING

    def apply_difficulty(self):
        rates = {"EASY": 0.2, "MEDIUM": 0.35, "HARD": 0.6}
        rate = rates.get(self.difficulty, 0.35)
        for p in self.players:
            p.bubble.shrink_rate = rate

    def cycle_difficulty(self):
        diffs = ["EASY", "MEDIUM", "HARD"]
        idx = diffs.index(self.difficulty)
        self.difficulty = diffs[(idx + 1) % len(diffs)]
        self.apply_difficulty()

    def switch_mode(self, num_players: int):
        """Switch number of players in place."""
        self.num_players = num_players
        colors = [P1_COLOR, P2_COLOR]
        self.players = [
            Player(i, colors[i], self.max_bubble_size)
            for i in range(num_players)
        ]
        self.reset()
        self.set_bubble_positions(self.screen_w, self.screen_h)

    def handle_click(self, x: int, y: int):
        """Detect sidebar button clicks based on coordinates."""
        w = self.screen_w
        # Sidebar bounds: x: w-170 to w-20
        if w - 170 <= x <= w - 20:
            # Mode Button: y: 180 to 230
            if 180 <= y <= 230:
                self.pending_action = "mode"
            # Start/Pause Button: y: 245 to 295
            elif 245 <= y <= 295:
                self.pending_action = "start_pause"
            # Reset Button: y: 310 to 360
            elif 310 <= y <= 360:
                self.pending_action = "reset"
            # Difficulty Button: y: 375 to 425
            elif 375 <= y <= 425:
                self.pending_action = "difficulty"
            # Quit Button: y: 440 to 490
            elif 440 <= y <= 490:
                self.pending_action = "quit"

    def get_and_clear_action(self) -> str | None:
        """Returns clicked action and clears the trigger."""
        action = self.pending_action
        self.pending_action = None
        return action

    def assign_faces(self, face_data: list, screen_w: int):
        """
        Assign face landmarks to players. 
        Resets player active status every frame to prevent stuck bubbles when tracking is lost.
        """
        # Set all active states to False for the current frame
        for p in self.players:
            p.is_active = False

        if len(face_data) == 0:
            return

        if self.num_players == 1:
            # Track the face closest to the screen's center
            sorted_by_center = sorted(face_data, key=lambda f: abs(f[2] - screen_w / 2))
            lm, metrics, cx = sorted_by_center[0]
            self.players[0].process_face(lm, metrics)
            return

        # 2-player mode:
        # If there are at least 2 faces, sort them by x-coordinate (left to right)
        # and assign the leftmost to P1, rightmost to P2.
        if len(face_data) >= 2:
            sorted_faces = sorted(face_data, key=lambda f: f[2])
            self.players[0].process_face(sorted_faces[0][0], sorted_faces[0][1])
            self.players[1].process_face(sorted_faces[-1][0], sorted_faces[-1][1])
        else:
            # Only 1 face detected. Assign to P1 or P2 depending on screen half
            face = face_data[0]
            cx = face[2]
            if cx < screen_w / 2:
                self.players[0].process_face(face[0], face[1])
            else:
                self.players[1].process_face(face[0], face[1])

    def update(self) -> bool:
        """
        Main update called every frame.
        Returns True if we just entered WINNER state.
        """
        if self.state == GameState.COUNTDOWN:
            self._update_countdown()
            return False

        if self.state == GameState.PLAYING:
            return self._update_playing()

        if self.state == GameState.PAUSED:
            return False

        if self.state == GameState.WINNER:
            self._update_winner_timeout()
            return False

        return False

    # ── Countdown ─────────────────────────────────────────────────────────────

    def _update_countdown(self):
        elapsed = time.time() - self._countdown_start
        if elapsed >= self.COUNTDOWN_SECS:
            self.state = GameState.PLAYING
            # Reset blow detectors so first frame doesn't misfire
            for p in self.players:
                p.blow_detector.reset()

    def get_countdown_value(self) -> int | None:
        """Returns remaining countdown int (3,2,1) or None."""
        if self.state != GameState.COUNTDOWN:
            return None
        elapsed = time.time() - self._countdown_start
        remaining = self.COUNTDOWN_SECS - int(elapsed)
        if remaining <= 0:
            return None
        return remaining

    # ── Playing ───────────────────────────────────────────────────────────────

    def _update_playing(self) -> bool:
        newly_won = False
        for player in self.players:
            player.update()
            just_popped = player.bubble.update()
            if just_popped and self.winner_id is None:
                self.winner_id = player.player_id
                player.bubble.winner = True
                self.state = GameState.WINNER
                self._winner_start = time.time()
                newly_won = True
        return newly_won

    # ── Winner screen timeout ─────────────────────────────────────────────────

    def _update_winner_timeout(self):
        # Keep updating bubble pop particles
        for player in self.players:
            player.bubble.update()

        if time.time() - self._winner_start >= self.WIN_HOLD_SECS:
            self._reset_players()
            self.state = GameState.MENU
            self.winner_id = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_players(self):
        for p in self.players:
            p.reset()

    @property
    def is_playing(self) -> bool:
        return self.state == GameState.PLAYING

    @property
    def is_menu(self) -> bool:
        return self.state == GameState.MENU

    @property
    def is_winner(self) -> bool:
        return self.state == GameState.WINNER

    @property
    def is_countdown(self) -> bool:
        return self.state == GameState.COUNTDOWN

    def set_bubble_positions(self, frame_w: int, frame_h: int):
        """
        Set where each player's bubble appears on screen.
        If player is active and mouth center is tracked, bubble stays locked on mouth.
        Otherwise, holds the last known position to prevent visual jumps, falling back to
        defaults during MENU/COUNTDOWN or initially.
        """
        self.screen_w = frame_w
        self.screen_h = frame_h
        margin_y = int(frame_h * 0.38)
        
        # Reset to screen centers during menus or countdown
        force_default = self.state in (GameState.MENU, GameState.COUNTDOWN)
        
        # Player 0
        if not force_default and self.players[0].is_active and self.players[0].mouth_center is not None:
            self.players[0].bubble.screen_pos = self.players[0].mouth_center
        else:
            if force_default or self.players[0].bubble.screen_pos == (0, 0):
                self.players[0].bubble.screen_pos = (frame_w // 2, margin_y) if self.num_players == 1 else (frame_w // 4, margin_y)

        # Player 1
        if self.num_players == 2:
            if not force_default and self.players[1].is_active and self.players[1].mouth_center is not None:
                self.players[1].bubble.screen_pos = self.players[1].mouth_center
            else:
                if force_default or self.players[1].bubble.screen_pos == (0, 0):
                    self.players[1].bubble.screen_pos = (3 * frame_w // 4, margin_y)
