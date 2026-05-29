from state import AppState
from screen_match import ScreenMatcher
from controller import DetectorController
from gui import SoundDetectorGUI

if __name__ == "__main__":
    state = AppState()
    matcher = ScreenMatcher()
    controller = DetectorController(state, matcher)
    SoundDetectorGUI(state, controller)