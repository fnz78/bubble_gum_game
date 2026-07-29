import sys
import os
import time
import streamlit as st
import cv2
import av

# Add src/ to python path so we can import game components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

from face_tracker import FaceTracker
from game_logic import GameLogic, GameState
from ui_drawing import (
    draw_background, draw_menu, draw_countdown, draw_hud,
    draw_player_labels, draw_paused_overlay, draw_winner, draw_fps
)
from utils import lerp

class BubbleGumVideoProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.tracker = FaceTracker()
        except Exception as e:
            st.error(f"Failed to initialize face tracker: {e}")
            self.tracker = None
            
        self.game = GameLogic(num_players=1)
        self.game.set_bubble_positions(640, 480)
        self.t = 0.0
        self.prev_time = time.time()
        self.fps_disp = 30.0

        self.pending_action = None
        self.pending_mode = None
        self.pending_difficulty = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        self.fps_disp = lerp(self.fps_disp, 1.0 / max(dt, 1e-4), 0.1)
        self.t += dt

        # Apply thread actions
        if self.pending_action == "start_pause":
            if self.game.state in (GameState.MENU, GameState.WINNER):
                self.game.start_game()
            else:
                self.game.toggle_pause()
            self.pending_action = None
        elif self.pending_action == "reset":
            self.game.reset()
            self.pending_action = None

        if self.pending_mode is not None:
            self.game.switch_mode(self.pending_mode)
            self.pending_mode = None

        if self.pending_difficulty is not None:
            self.game.difficulty = self.pending_difficulty
            self.game.apply_difficulty()
            self.pending_difficulty = None

        # Face tracking with error handling
        face_data = []
        if self.tracker:
            try:
                results = self.tracker.process(img)
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks:
                        metrics = self.tracker.get_mouth_metrics(lm, w, h)
                        cx = self.tracker.get_face_center_x(lm, w, h)
                        face_data.append((lm, metrics, cx))
                        self.tracker.draw_mouth_landmarks(img, metrics, color=(0, 255, 120))
            except Exception as e:
                # Silently handle tracking errors
                pass
        
        self.game.assign_faces(face_data, w)
        self.game.update()

        # Render layout based on state
        draw_background(img)
        state = self.game.state

        if state == GameState.MENU:
            draw_menu(img, self.game.num_players, self.t)

        elif state == GameState.COUNTDOWN:
            self.game.set_bubble_positions(w, h)
            for p in self.game.players:
                p.bubble.draw(img)
            cv_val = self.game.get_countdown_value()
            draw_countdown(img, cv_val if cv_val else 0, self.t)

        elif state == GameState.PLAYING:
            self.game.set_bubble_positions(w, h)
            for p in self.game.players:
                p.bubble.draw(img)
            draw_hud(img, self.game, self.t)
            draw_player_labels(img, self.game.num_players)

        elif state == GameState.PAUSED:
            self.game.set_bubble_positions(w, h)
            for p in self.game.players:
                p.bubble.draw(img)
            draw_hud(img, self.game, self.t)
            draw_paused_overlay(img, self.t)

        elif state == GameState.WINNER:
            self.game.set_bubble_positions(w, h)
            for p in self.game.players:
                p.bubble.draw(img)
            draw_winner(img, self.game.winner_id, self.t)

        draw_fps(img, self.fps_disp)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Bubble Gum Blow Challenge",
        page_icon="🎈",
        layout="centered"
    )

    st.title("🎈 Bubble Gum Blow Challenge 💨")
    st.markdown(
        """
        Play the **Bubble Gum Blow Challenge** directly in your browser using your webcam!
        
        ### 🎮 How to Play:
        1. Choose your **Mode** and **Difficulty** in the sidebar.
        2. Click **Start** inside the WebRTC player component below to request webcam access.
        3. Click **Start/Pause Game** or **Reset Game** in the sidebar to control play state.
        4. Open your mouth wide, then close it quickly to "blow" and inflate your bubble. Don't let it shrink!
        """
    )

    # Sidebar controls
    st.sidebar.header("Controls & Settings")
    mode = st.sidebar.selectbox("Players", ["1-Player", "2-Player"])
    difficulty = st.sidebar.selectbox("Difficulty", ["EASY", "MEDIUM", "HARD"], index=1)
    
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("Start/Pause")
    reset_btn = col2.button("Reset")

    # RTC Configuration with public Google STUN servers
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="bubblegum",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=BubbleGumVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Sync UI buttons to running video processor thread
    if ctx and ctx.video_processor:
        target_players = 1 if mode == "1-Player" else 2
        if ctx.video_processor.game.num_players != target_players:
            ctx.video_processor.pending_mode = target_players

        if ctx.video_processor.game.difficulty != difficulty:
            ctx.video_processor.pending_difficulty = difficulty

        if start_btn:
            ctx.video_processor.pending_action = "start_pause"
        if reset_btn:
            ctx.video_processor.pending_action = "reset"

if __name__ == "__main__":
    main()