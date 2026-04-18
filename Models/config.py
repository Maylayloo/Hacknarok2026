from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

action_to_number = {
    'BIG_LEFT': 0,
    'BIG_RIGHT': 1,
    'CINEMA_MODE': 2,
    'CLICK': 3,
    'IDLE': 4,
    'OPEN_PALM': 5,
    'SITE_LEFT': 6,
    'SITE_RIGHT': 7,
    'SKROLL_DOWN': 8,
    'SKROLL_UP': 9,
    'SMALL_LEFT': 10,
    'SMALL_RIGHT': 11,
    'SWIPE_LEFT': 12,
    'SWIPE_RIGHT': 13,
    'VIDEO': 14,
    'VOLUME_DOWN': 15,
    'VOLUME_UP': 16
}

number_to_action = {
    0: 'BIG_LEFT',
    1: 'BIG_RIGHT',
    2: 'CINEMA_MODE',
    3: 'CLICK',
    4: 'IDLE',
    5: 'OPEN_PALM',
    6: 'SITE_LEFT',
    7: 'SITE_RIGHT',
    8: 'SKROLL_DOWN',
    9: 'SKROLL_UP',
    10: 'SMALL_LEFT',
    11: 'SMALL_RIGHT',
    12: 'SWIPE_LEFT',
    13: 'SWIPE_RIGHT',
    14: 'VIDEO',
    15: 'VOLUME_DOWN',
    16: 'VOLUME_UP'
}
