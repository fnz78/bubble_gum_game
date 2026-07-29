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
from utils import lerp
from ui_drawing import (
    draw_background, draw_menu, draw_countdown, draw_hud,
    draw_player_labels, draw_control_panel, draw_paused_overlay,
    draw_winner, draw_fps
)

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_NAME   = "Bubble Gum Blow Challenge"
CAM_W, CAM_H  = 1280, 720
MAX_BUBBLE_PX = 130


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
