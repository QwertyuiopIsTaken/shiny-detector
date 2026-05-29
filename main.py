from state import AppState
from detector import ShinyDetector
from screen_match import ScreenMatcher
from controller import DetectorController
from gui import SoundDetectorGUI

if __name__ == "__main__":
    state = AppState()
    detector = ShinyDetector(state)
    matcher = ScreenMatcher()
    controller = DetectorController(state, detector, matcher)
    SoundDetectorGUI(state, controller)