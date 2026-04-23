from __future__ import annotations
from typing import TYPE_CHECKING

import os
import sys
import importlib
from dotenv import load_dotenv

if TYPE_CHECKING:
    from controller import Motor, Robot


def get_webots_robot() -> Robot:
    load_dotenv()
    """Resolve the WEBOTS_HOME path, add it to sys.path, and return the Robot class."""
    webots_home = os.getenv("WEBOTS_HOME")
    if not webots_home:
        raise ValueError("WEBOTS_HOME not found in .env file!")

    controller_path = os.path.join(webots_home, "lib", "controller", "python")
    if controller_path not in sys.path:
        sys.path.append(controller_path)

    controller = importlib.import_module("controller")
    return controller.Robot()