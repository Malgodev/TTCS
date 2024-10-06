import time
from inputs import get_key, KeyCode

class XboxEmulator:
    def __init__(self):
        self.key_map = {
            'A': KeyCode.BUTTON_A,
            'B': KeyCode.BUTTON_B,
            'X': KeyCode.BUTTON_X,
            'Y': KeyCode.BUTTON_Y,
            'LB': KeyCode.BUTTON_L,
            'RB': KeyCode.BUTTON_R,
            'LT': KeyCode.BUTTON_THUMBL,
            'RT': KeyCode.BUTTON_THUMBR,
            'SELECT': KeyCode.BUTTON_SELECT,
            'START': KeyCode.BUTTON_START,
        }

    def press_btn(self, key):
        key_code = self.key_map.get(key)
        if key_code:
            get_key(key_code, 1)
            get_key(key_code, 0)
            time.sleep(0.1)        

    def emulate_controller(self):
        self.press_btn('A')


if __name__ == '__main__':
    controller_emulator = XboxEmulator()
    while(True):
        print("test")
        controller_emulator.emulate_controller()
        time.sleep(5)