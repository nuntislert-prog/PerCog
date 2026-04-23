from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, Keyboard

class WebotsKeyboard:
    def __init__(self, robot: "Robot"):
        timestep = int(robot.getBasicTimeStep())
        self.keyboard = cast("Keyboard", robot.getKeyboard())
        self.keyboard.enable(timestep)

    def getKey(self):
        return self.keyboard.getKey()