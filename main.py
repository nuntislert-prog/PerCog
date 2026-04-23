import cv2

from utils.keyboard import WebotsKeyboard
from utils.robot import get_webots_robot

from mapping.kinematics import DiffDriveOdometry, calculate_diff_drive_velocities
from mapping.grid import OccupancyGrid

lin_vel = 0.5  # m/s
ang_vel = 2.0  # rad/s

def get_compass_sensor(robot):
    compass = robot.getDevice("compass")
    timestep = int(robot.getBasicTimeStep())
    compass.enable(timestep)
    return compass

def run_sim() -> None:
    robot = get_webots_robot()
    timestep = int(robot.getBasicTimeStep())

    # Motor
    LMotor = robot.getDevice("left wheel motor")
    LMotor.setPosition(float("inf"))
    LMotor.setVelocity(0.0)
    RMotor = robot.getDevice("right wheel motor")
    RMotor.setPosition(float("inf"))
    RMotor.setVelocity(0.0)
    wheels = [LMotor, RMotor]

    # Encoder
    LEncoder = robot.getDevice("left wheel sensor")
    REncoder = robot.getDevice("right wheel sensor")
    LEncoder.enable(timestep)
    REncoder.enable(timestep)

    # Compass
    compass = get_compass_sensor(robot)
    compass.getValues()
    
    # Lidar
    lidar = robot.getDevice("lidar")
    lidar.enable(timestep)
    lidar_fov = lidar.getFov()
    lidar_max_range = lidar.getMaxRange()

    # Keyboard
    keyboard = WebotsKeyboard(robot)

    # Odometry and grid
    odometry = DiffDriveOdometry(LEncoder, REncoder, compass)
    grid = OccupancyGrid(world_min=(-10.0, -10.0), world_max=(5.0, 5.0), resolution=0.02)

    while robot.step(timestep) != -1:
        velocties = [0, 0]

        key = keyboard.getKey()
        while key != -1:
            char = chr(key) if 0 < key < 128 else ""
            if char in ("W", "w"):
                velocties = [15, 15]
            elif char in ("S", "s"):
                velocties = [-15, -15]
            elif char in ("A", "a"):
                velocties = [-2, 2]
            elif char in ("D", "d"):
                velocties = [2, -2]
            key = keyboard.getKey()

        
        for wheel, vel in zip(wheels, velocties):
            wheel.setVelocity(float(vel))
        odometry.update()
        pose = odometry.get_pose()

        ranges = lidar.getRangeImage()
        if ranges:
            grid.update(pose, list(ranges), lidar_fov, lidar_max_range)

        map_img = grid.render()
        cv2.imshow("SLAM Map", map_img)
        cv2.waitKey(1)

    for wheel in wheels:
        wheel.setVelocity(0.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_sim()