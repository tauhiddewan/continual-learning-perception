YCB_CLASSES = {
    0: "background",
    1: "master_chef_can",
    2: "cracker_box",
    3: "sugar_box",
    4: "tomato_soup_can",
    5: "mustard_bottle",
    6: "tuna_fish_can",
    7: "pudding_box",
    8: "gelatin_box",
    9: "potted_meat_can",
    10: "banana",
    11: "pitcher_base",
    12: "bleach_cleanser",
    13: "bowl",
    14: "mug",
    15: "power_drill",
    16: "wood_block",
    17: "scissors",
    18: "large_marker",
    19: "large_clamp",
    20: "extra_large_clamp",
    21: "foam_brick",
}

# simple split: 10 base classes, 11 new classes
BASE_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NEW_CLASS_IDS  = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

NUM_CLASSES = 22  # 0..21
IGNORE_INDEX = 255
