"""
main.py - Bubble Gum Blow Challenge  (Clean UI Redesign)
Controls: 1/2=mode  S=start  R=reset  Q/ESC=quit
"""

import sys, os, math, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_tracker import FaceTracker
from game_logic  import GameLogic, GameState
from utils       import (
    draw_bubble, draw_pop_particles, create_pop_particles,
    update_pop_particles, lerp, lerp_color, rainbow_color,
    P1_COLOR, P2_COLOR, WHITE, GREEN_NEON, YELLOW_NEON, PINK_LIGHT,
    RED_NEON, hsv_to_bgr, CYAN_NEON
)

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_NAME   = "Bubble Gum Blow Challenge"
CAM_W, CAM_H  = 1280, 720
MAX_BUBBLE_PX = 130

# ── Design tokens ──────────────────────────────────────────────────────────────
DARK_BG     = (20, 20, 20)        # neutral dark gray
PANEL_BG    = (35, 25, 55)        # card background
PANEL_EDGE  = (90, 60, 130)       # card border
ACCENT_PINK = (210, 120, 255)
ACCENT_BLUE = (255, 190,  80)
TEXT_MUTED  = (160, 140, 180)
TEXT_DIM    = (100,  90, 115)
FONT_MAIN   = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_SIMPLEX


# ── Low-level drawing primitives ───────────────────────────────────────────────

def _text_size(text, font, scale, thickness):
    (w, h), bl = cv2.getTextSize(text, font, scale, thickness)
    return w, h + bl

def put_text(frame, text, pos, font=FONT_MAIN, scale=0.7,
             color=WHITE, thickness=1, anchor="tl"):
    tw, th = _text_size(text, font, scale, thickness)
    x, y   = pos
    if "c" in anchor: x -= tw // 2
    if "r" in anchor: x -= tw
    if "b" in anchor: y -= th
    # subtle shadow
    cv2.putText(frame, text, (x+1, y+1), font, scale,
                (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),   font, scale,
                color, thickness, cv2.LINE_AA)

def put_centered(frame, text, cy, font=FONT_MAIN, scale=1.0,
                 color=WHITE, thickness=2):
    w = frame.shape[1]
    tw, _ = _text_size(text, font, scale, thickness)
    put_text(frame, text, (w//2 - tw//2, cy), font, scale, color, thickness)

def draw_rect(frame, x, y, w, h, color, alpha=1.0, radius=0, border_color=None):
    """Draw a filled rectangle, optionally semi-transparent."""
    if alpha < 1.0:
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
    if border_color:
        cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 1, cv2.LINE_AA)

def draw_panel(frame, x, y, w, h, alpha=0.82):
    """Frosted glass-style panel."""
    draw_rect(frame, x, y, w, h, PANEL_BG, alpha, border_color=PANEL_EDGE)

def draw_pill_bar(frame, x, y, w, h, value, max_val, fill_color,
                  bg=(45, 35, 65), label="", label_color=None):
    """Rounded progress bar."""
    frac    = min(value / max(max_val, 1e-6), 1.0)
    filled  = int(w * frac)
    r       = h // 2                  # radius for rounded ends

    # background
    cv2.rectangle(frame, (x+r, y), (x+w-r, y+h), bg, -1)
    cv2.circle(frame, (x+r,   y+r), r, bg, -1)
    cv2.circle(frame, (x+w-r, y+r), r, bg, -1)

    # fill
    if filled > r*2:
        cv2.rectangle(frame, (x+r, y), (x+filled-r, y+h), fill_color, -1)
        cv2.circle(frame, (x+r,       y+r), r, fill_color, -1)
        cv2.circle(frame, (x+filled-r,y+r), r, fill_color, -1)
    elif filled > 0:
        cv2.circle(frame, (x+r, y+r), r, fill_color, -1)

    # label
    if label:
        lc = label_color or TEXT_MUTED
        tw, th = _text_size(label, FONT_MONO, 0.38, 1)
        lx = x + (w - tw) // 2
        ly = y + (h + th) // 2 - 1
        cv2.putText(frame, label, (lx, ly), FONT_MONO,
                    0.38, lc, 1, cv2.LINE_AA)


# ── Background ─────────────────────────────────────────────────────────────────

def draw_background(frame):
    """Clean dark vignette overlay — vectorized, fast."""
    h, w = frame.shape[:2]
    # Semi-transparent dark blend (0.20 opacity for clear, natural look)
    ov = np.full_like(frame, DARK_BG, dtype=np.uint8)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    # Vignette via numpy: darker at corners
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    mask = np.clip((xs**2 + ys**2) * 0.55, 0.0, 1.0)          # (H, W)
    vignette = (mask[:, :, np.newaxis] * 40).astype(np.uint8)  # (H, W, 1)
    vignette = np.broadcast_to(vignette, frame.shape)           # (H, W, 3)
    frame[:] = cv2.subtract(frame, vignette.copy())


# ── Menu ───────────────────────────────────────────────────────────────────────

def draw_menu(frame, num_players, t):
    h, w = frame.shape[:2]

    # ── center card ──────────────────────────────────────────────────────────
    cw, ch = 520, 360
    cx = (w - cw) // 2
    cy = (h - ch) // 2 - 20
    draw_panel(frame, cx, cy, cw, ch, alpha=0.88)

    # thin rainbow top border
    for xi in range(cw):
        col = rainbow_color((t*0.25 + xi/cw) % 1.0)
        frame[cy, cx+xi] = col
        frame[cy+1, cx+xi] = tuple(int(c*0.5) for c in col)

    # ── title ────────────────────────────────────────────────────────────────
    title_y = cy + 68
    put_centered(frame, "BUBBLE GUM", title_y,
                 scale=1.9, color=rainbow_color((t*0.3) % 1.0), thickness=3)
    put_centered(frame, "BLOW CHALLENGE", title_y + 54,
                 scale=1.1, color=ACCENT_PINK, thickness=2)

    # divider
    div_y = cy + 148
    cv2.line(frame, (cx+30, div_y), (cx+cw-30, div_y), PANEL_EDGE, 1)

    # ── mode badge ───────────────────────────────────────────────────────────
    mode_badge_x = w//2 - 80
    mode_badge_y = div_y + 16
    badge_col    = P1_COLOR if num_players == 1 else P2_COLOR
    draw_rect(frame, mode_badge_x, mode_badge_y, 160, 32,
              badge_col, alpha=0.25, border_color=badge_col)
    mode_lbl = f"  {'1-PLAYER' if num_players==1 else '2-PLAYER'}  "
    put_text(frame, mode_lbl,
             (mode_badge_x + 10, mode_badge_y + 22),
             font=FONT_MONO, scale=0.65, color=badge_col, thickness=1)

    # ── key hints ────────────────────────────────────────────────────────────
    keys = [
        ("[S]", "Start Game",    GREEN_NEON),
        ("[1] [2]", "Switch Mode", YELLOW_NEON),
        ("[D]", "Difficulty",     CYAN_NEON),
        ("[R]", "Reset",         TEXT_MUTED),
        ("[Q]", "Quit",          TEXT_MUTED),
    ]
    row_y = div_y + 60
    col1 = cx + 50
    col2 = cx + 140
    for short, label, col in keys:
        put_text(frame, short, (col1, row_y),
                 font=FONT_MONO, scale=0.58, color=col, thickness=1)
        put_text(frame, label, (col2, row_y),
                 font=FONT_MONO, scale=0.58, color=TEXT_MUTED, thickness=1)
        row_y += 28

    # ── bottom bubbles (decorative) ──────────────────────────────────────────
    bubble_data = [
        (0.12, 0.85, 18), (0.28, 0.90, 14), (0.50, 0.87, 22),
        (0.72, 0.91, 16), (0.88, 0.84, 20),
    ]
    for bx_frac, by_frac, base_r in bubble_data:
        idx  = bubble_data.index((bx_frac, by_frac, base_r))
        phase = t * 1.1 + idx * 1.4
        bx  = int(w * bx_frac + math.sin(phase)       * 18)
        by  = int(h * by_frac + math.cos(phase * 0.8) * 12)
        br  = int(base_r + 4 * math.sin(phase * 1.6))
        col = rainbow_color((t * 0.2 + idx * 0.18) % 1.0)
        draw_bubble(frame, (bx, by), br, col)


# ── HUD (in-game) ──────────────────────────────────────────────────────────────

def draw_hud(frame, game, t):
    h, w = frame.shape[:2]

    for player in game.players:
        pid    = player.player_id
        color  = P1_COLOR if pid == 0 else P2_COLOR
        bubble = player.bubble

        # ── panel geometry ───────────────────────────────────────────────────
        bar_w  = 280
        margin = 24
        panel_h = 90
        panel_w = bar_w + 20
        px = margin if pid == 0 else w - margin - panel_w
        py = margin

        draw_panel(frame, px, py, panel_w, panel_h, alpha=0.80)
        # colored left stripe
        cv2.rectangle(frame, (px, py), (px+4, py+panel_h), color, -1)

        # ── player label ─────────────────────────────────────────────────────
        label = f"P{pid+1}  {int(bubble.fraction*100)}%"
        put_text(frame, label, (px+14, py+20),
                 font=FONT_MONO, scale=0.62, color=color, thickness=1)

        # ── bubble size bar ───────────────────────────────────────────────────
        draw_pill_bar(frame,
                      px+10, py+30, bar_w, 16,
                      bubble.size, bubble.max_size, color,
                      label="BUBBLE SIZE")

        # ── blow / openness bars ──────────────────────────────────────────────
        half = (bar_w - 6) // 2
        # openness
        draw_pill_bar(frame,
                      px+10, py+57, half, 12,
                      player.openness, 0.55,
                      (100, 180, 255), label="OPEN")
        # blow
        blow_col = GREEN_NEON if player.blow_intensity > 0.05 else (60, 60, 80)
        draw_pill_bar(frame,
                      px+10+half+6, py+57, half, 12,
                      player.blow_intensity, 1.0,
                      blow_col, label="BLOW")

    # ── 2-player divider ──────────────────────────────────────────────────────
    if game.num_players == 2:
        cv2.line(frame, (w//2, 10), (w//2, h-10),
                 PANEL_EDGE, 1, cv2.LINE_AA)

    # ── tip bar at bottom ─────────────────────────────────────────────────────
    tip_h = 30
    draw_panel(frame, 0, h-tip_h, w, tip_h, alpha=0.70)
    tip = "Open mouth then close to BLOW   |   R=Reset   Q=Quit"
    put_centered(frame, tip, h-8, font=FONT_MONO,
                 scale=0.50, color=TEXT_MUTED, thickness=1)


# ── Countdown ──────────────────────────────────────────────────────────────────

def draw_countdown(frame, value, t):
    h, w = frame.shape[:2]
    num_str = str(value) if value > 0 else "GO !"
    col     = YELLOW_NEON if value > 0 else GREEN_NEON
    pulse   = 1.0 + 0.12 * math.sin(t * 9)
    scale   = 4.2 * pulse

    # dark pill behind number
    tw, th = _text_size(num_str, FONT_MAIN, scale, 6)
    px, py = (w - tw)//2 - 24, h//2 - th//2 - 20
    draw_panel(frame, px, py, tw+48, th+40, alpha=0.75)

    put_centered(frame, num_str, h//2 + 30,
                 font=FONT_MAIN, scale=scale, color=col, thickness=6)

    sub = "GET READY TO BLOW !" if value > 0 else "BLOW NOW !"
    put_centered(frame, sub, h//2 + 100,
                 font=FONT_MONO, scale=0.85, color=WHITE, thickness=1)


# ── Winner ─────────────────────────────────────────────────────────────────────

def draw_winner(frame, winner_id, t):
    h, w = frame.shape[:2]
    color = P1_COLOR if winner_id == 0 else P2_COLOR

    # full overlay
    ov = frame.copy()
    ov[:] = DARK_BG
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

    # rainbow ribbon
    rib_y1, rib_y2 = h//2-80, h//2+80
    for y in range(rib_y1, rib_y2):
        col = rainbow_color((t*0.4 + y/h) % 1.0)
        ov2 = frame.copy()
        cv2.line(ov2, (0,y), (w,y), col, 1)
        cv2.addWeighted(ov2, 0.025, frame, 0.975, 0, frame)

    # center card
    cw, ch = 480, 200
    cx = (w-cw)//2;  cy = (h-ch)//2
    draw_panel(frame, cx, cy, cw, ch, alpha=0.88)
    # top rainbow strip
    for xi in range(cw):
        col = rainbow_color((t*0.3 + xi/cw) % 1.0)
        frame[cy, cx+xi] = col

    put_centered(frame, "WINNER !",
                 cy+62, scale=2.0,
                 color=rainbow_color(t % 1.0), thickness=3)
    pname = f"PLAYER  {winner_id+1}"
    put_centered(frame, pname, cy+118,
                 scale=1.5, color=color, thickness=3)

    # hint
    draw_panel(frame, 0, h-34, w, 34, alpha=0.70)
    put_centered(frame, "S = Play Again     R = Menu     Q = Quit",
                 h-10, font=FONT_MONO, scale=0.55, color=TEXT_MUTED, thickness=1)


# ── Face assignment labels ─────────────────────────────────────────────────────

def draw_player_labels(frame, num_players):
    if num_players < 2:
        return
    h, w = frame.shape[:2]
    lbl_h = 28
    draw_panel(frame, 0, h-lbl_h-30, 110, lbl_h, alpha=0.75)
    put_text(frame, "P1 (LEFT)", (8, h-38),
             font=FONT_MONO, scale=0.55, color=P1_COLOR, thickness=1)
    draw_panel(frame, w-114, h-lbl_h-30, 110, lbl_h, alpha=0.75)
    put_text(frame, "P2 (RIGHT)", (w-110, h-38),
             font=FONT_MONO, scale=0.55, color=P2_COLOR, thickness=1)


def draw_control_panel(frame, game):
    h, w = frame.shape[:2]
    # Panel background on the right edge
    px = w - 180
    py = 150
    pw = 170
    ph = 365
    draw_panel(frame, px, py, pw, ph, alpha=0.85)
    
    # Title
    put_text(frame, "CONTROLS", (px + 28, py + 22), font=FONT_MAIN, scale=0.55, color=ACCENT_PINK, thickness=2)
    cv2.line(frame, (px + 15, py + 30), (px + pw - 15, py + 30), PANEL_EDGE, 1)

    # Button labels based on states
    mode_label = "2-PLAYER" if game.num_players == 1 else "1-PLAYER"
    
    start_label = "START [S]"
    if game.state == GameState.PLAYING:
        start_label = "PAUSE [S]"
    elif game.state == GameState.PAUSED:
        start_label = "RESUME [S]"
    elif game.state == GameState.COUNTDOWN:
        start_label = "READY..."

    diff_label = f"DIFF: {game.difficulty} [D]"

    btn_configs = [
        (mode_label, 180, 230, ACCENT_BLUE),
        (start_label, 245, 295, GREEN_NEON),
        ("RESET [R]", 310, 360, YELLOW_NEON),
        (diff_label, 375, 425, CYAN_NEON),
        ("QUIT [Q]", 440, 490, RED_NEON)
    ]

    for label, y_start, y_end, color in btn_configs:
        bx1, bx2 = w - 170, w - 20
        # Hover check
        is_hovered = (bx1 <= game.mouse_x <= bx2) and (y_start <= game.mouse_y <= y_end)
        
        # Draw background rect
        bg_alpha = 0.35 if is_hovered else 0.12
        border_col = color if is_hovered else PANEL_EDGE
        draw_rect(frame, bx1, y_start, bx2 - bx1, y_end - y_start, color, alpha=bg_alpha, border_color=border_col)
        
        # Center the text vertically
        ty = y_start + 32
        tx = bx1 + 18
        
        text_color = WHITE if is_hovered else TEXT_MUTED
        put_text(frame, label, (tx, ty), font=FONT_MONO, scale=0.45, color=text_color, thickness=1)


def draw_paused_overlay(frame, t):
    h, w = frame.shape[:2]
    cw, ch = 320, 110
    cx = (w - cw) // 2
    cy = (h - ch) // 2
    draw_panel(frame, cx, cy, cw, ch, alpha=0.92)
    
    # Thin rainbow top border
    for xi in range(cw):
        col = rainbow_color((t * 0.25 + xi / cw) % 1.0)
        frame[cy, cx + xi] = col
        frame[cy+1, cx+xi] = tuple(int(c * 0.5) for c in col)
        
    put_centered(frame, "GAME PAUSED", cy + 45, scale=1.1, color=YELLOW_NEON, thickness=2)
    put_centered(frame, "Press 'S' or click RESUME", cy + 80, font=FONT_MONO, scale=0.48, color=WHITE, thickness=1)


def draw_fps(frame, fps):
    put_text(frame, f"{fps:.0f} fps",
             (frame.shape[1]-70, frame.shape[0]-8),
             font=FONT_MONO, scale=0.42, color=TEXT_DIM, thickness=1)


def mouse_callback(event, x, y, flags, param):
    game = param
    if event == cv2.EVENT_MOUSEMOVE:
        game.mouse_x = x
        game.mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        game.handle_click(x, y)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = FaceTracker(max_faces=2)

    num_players = 1
    game = GameLogic(num_players=num_players, max_bubble_size=MAX_BUBBLE_PX)
    game.set_bubble_positions(actual_w, actual_h)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, game)

    t         = 0.0
    prev_time = time.time()
    fps_disp  = 30.0

    print("* Bubble Gum Blow Challenge  |  1/2=mode  S=start  R=reset  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        now       = time.time()
        dt        = now - prev_time
        prev_time = now
        fps_disp  = lerp(fps_disp, 1.0 / max(dt, 1e-4), 0.1)
        t        += dt

        # ── Mouse Actions ─────────────────────────────────────────────────────
        action = game.get_and_clear_action()
        if action == "quit":
            break
        elif action == "start_pause":
            if game.state in (GameState.MENU, GameState.WINNER):
                game.start_game()
            else:
                game.toggle_pause()
        elif action == "reset":
            game.reset()
        elif action == "mode":
            new_mode = 2 if game.num_players == 1 else 1
            game.switch_mode(new_mode)
            num_players = new_mode
        elif action == "difficulty":
            game.cycle_difficulty()

        # ── Face detection ────────────────────────────────────────────────────
        results   = tracker.process(frame)
        face_data = []
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                metrics = tracker.get_mouth_metrics(lm, actual_w, actual_h)
                cx      = tracker.get_face_center_x(lm, actual_w, actual_h)
                face_data.append((lm, metrics, cx))
                # draw small mouth dots
                tracker.draw_mouth_landmarks(frame, metrics,
                                             color=(0, 255, 120))
        game.assign_faces(face_data, actual_w)

        # ── Update ────────────────────────────────────────────────────────────
        game.update()

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_background(frame)

        state = game.state

        if state == GameState.MENU:
            draw_menu(frame, num_players, t)

        elif state == GameState.COUNTDOWN:
            game.set_bubble_positions(actual_w, actual_h)
            for p in game.players:
                p.bubble.draw(frame)
            cv = game.get_countdown_value()
            draw_countdown(frame, cv if cv else 0, t)

        elif state == GameState.PLAYING:
            game.set_bubble_positions(actual_w, actual_h)
            for p in game.players:
                p.bubble.draw(frame)
            draw_hud(frame, game, t)
            draw_player_labels(frame, num_players)

        elif state == GameState.PAUSED:
            game.set_bubble_positions(actual_w, actual_h)
            for p in game.players:
                p.bubble.draw(frame)
            draw_hud(frame, game, t)
            draw_paused_overlay(frame, t)

        elif state == GameState.WINNER:
            game.set_bubble_positions(actual_w, actual_h)
            for p in game.players:
                p.bubble.draw(frame)
            draw_winner(frame, game.winner_id, t)

        # Draw sidebar control buttons
        draw_control_panel(frame, game)

        draw_fps(frame, fps_disp)
        cv2.imshow(WINDOW_NAME, frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            if game.state in (GameState.MENU, GameState.WINNER):
                game.start_game()
            else:
                game.toggle_pause()
        elif key in (ord('r'), ord('R')):
            game.reset()
        elif key == ord('1'):
            game.switch_mode(1)
            num_players = 1
        elif key == ord('2'):
            game.switch_mode(2)
            num_players = 2
        elif key in (ord('d'), ord('D')):
            game.cycle_difficulty()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Thanks for playing! *")


if __name__ == "__main__":
    main()
