import math

def discretize_x_coordinate(x):
    # Twenty possible values for x
    if x >= 1:
        return 1
    if x <= -1:
        return -1

    return math.ceil(x * 10) / 10


def discretize_y_coordinate(y):
    # Twenty possible values for y
    if y >= 2:
        return 2
    if y <= 0:
        return 0

    return math.ceil(y * 10) / 10


def discretize_x_speed(x):
    # Forty possible values for x_speed
    if x >= 2:
        return 2
    if x <= -2:
        return -2

    return math.ceil(x * 10) / 10

def discretize_y_speed(y):
    # Forty possible values for y_speed
    if y >= 2:
        return 2
    if y <= -2:
        return -2

    return math.ceil(y * 10) / 10


def discretize_angle(angle):
    # Forty possible values for angle
    if angle >= 2:
        return 2
    if angle <= -2:
        return -2
    return math.ceil(angle * 10) / 10


def discretize_angle_speed(angle_speed):
    # 10 possible values for angle speed
    if angle_speed >= 5:
        return 5
    if angle_speed <= -5:
        return -5
    return math.ceil(angle_speed)

def discretize_state(state):
    return [discretize_x_coordinate(state[0]),
            discretize_y_coordinate(state[1]),
            discretize_x_speed(state[2]),
            discretize_y_speed(state[3]),
            discretize_angle(state[4]),
            discretize_angle_speed(state[5]),
            state[6],
            state[7]]
