# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CocoDataset(CustomDataset):
    ## Coco format but, EgoObejcts dataset

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
    # PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
    #            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    #            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
    #            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
    #            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    #            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
    #            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
    #            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
    #            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
    #            (134, 134, 103), (145, 148, 174), (255, 208, 186),
    #            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
    #            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
    #            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
    #            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
    #            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
    #            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
    #            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
    #            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
    #            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
    #            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
    #            (191, 162, 208)]
    CLASSES = ('accordion', 'adhesive_tape', 'air_conditioner', 'air_fryer', 'air_purifier', 'airplane', 'alarm_clock', 'almond', 'alpaca', 'aluminium_foil', 'ambulance', 'ant', 'antelope', 'apple', 'apricot', 'armadillo', 'artichoke', 'arugula', 'avocado', 'axe', 'baby_monitor', 'backpack', 'bacon', 'badminton_birdie', 'badminton_racket', 'bagel', 'balance_beam', 'balloon', 'banana', 'band_aid', 'banjo', 'barge', 'barrel', 'baseball_bat', 'baseball_glove', 'basketball', 'bat_animal', 'bathroom_cabinet', 'bathtub', 'beaker', 'beans', 'bee', 'beef', 'beehive', 'beer', 'bell_pepper', 'belt', 'bench', 'bicycle', 'bicycle_helmet', 'bicycle_wheel', 'bidet', 'billboard', 'billiard_table', 'binoculars', 'blackberry', 'blanket', 'blender', 'blue_jay', 'blueberry', 'bok_choy', 'bomb', 'bonsai', 'book', 'bookcase', 'boot', 'bottle', 'bottle_opener', 'bow_and_arrow', 'bowl', 'bowling_equipment', 'box', 'box_of_macaroni_and_cheese', 'boxing_gloves', 'brassiere', 'bread', 'bridges', 'briefcase', 'broccoli', 'bronze_sculpture', 'brown_bear', 'brussel_sprouts', 'bull', 'burrito', 'bus', 'bust', 'butterfly', 'cabbage', 'cabinetry', 'cake', 'cake_stand', 'calculator', 'calendar', 'camel', 'camera', 'can_opener', 'canary', 'candle', 'candy', 'cannon', 'canoe', 'cantaloupe', 'carrot', 'cart', 'cashew', 'cassette_deck', 'castle', 'cat', 'cat_furniture', 'caterpillar', 'cattle', 'cauliflower', 'ceiling_fan', 'celery', 'cello', 'centipede', 'chainsaw', 'chair', 'chandelier', 'chard', 'cheese', 'cheetah', 'cherry', 'cherry_tomato', 'chest_of_drawers', 'chicken', 'chicken_breast', 'chime', 'chisel', 'chive', 'chopsticks', 'christmas_tree', 'closet', 'coat', 'cocktail', 'cocktail_shaker', 'coconut', 'coffee', 'coffee_cup', 'coffee_table', 'coffeemaker', 'coin', 'collard_green', 'common_fig', 'common_sunflower', 'computer_keyboard', 'computer_monitor', 'computer_mouse', 'condiment', 'convenience_store', 'cookie', 'cooking_spray', 'corded_phone', 'countertop', 'cowboy_hat', 'crab', 'cream', 'creamer', 'crib', 'cricket_ball', 'crocodile', 'croissant', 'crown', 'crutch', 'cucumber', 'cupboard', 'curtain', 'cutting_board', 'dagger', 'deep_fryer', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'dice', 'digital_clock', 'dinosaur', 'dishwasher', 'dog', 'dog_bed', 'doll', 'dolphin', 'door', 'door_handle', 'doughnut', 'dragonfly', 'drawer', 'dress', 'drill_tool', 'drinking_straw', 'drivers_license', 'drum', 'duck', 'dumbbell', 'eagle', 'earrings', 'egg_food', 'eggplant', 'elephant', 'endive', 'envelope', 'eraser', 'face_powder', 'facial_tissue_holder', 'falcon', 'fax', 'fedora', 'filing_cabinet', 'fire', 'fire_alarm', 'fire_hydrant', 'fire_truck', 'firearm', 'fireplace', 'firework', 'fishing_pole', 'flag', 'flashlight', 'floor_lamp', 'flowerpot', 'flute', 'flying_disc', 'food_processor', 'football', 'football_helmet', 'fork', 'fountain', 'fox', 'french_fries', 'french_horn', 'frisée', 'frog', 'fruit_juice', 'frying_pan', 'game_controller_pad', 'garden_asparagus', 'garlic', 'gas_stove', 'gift', 'ginger', 'giraffe', 'glasses', 'glove', 'goat', 'goggles', 'goldfish', 'golf_ball', 'golf_cart', 'gondola', 'goose', 'grape', 'grapefruit', 'grinder', 'ground_chicken', 'ground_turkey', 'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger', 'hammer', 'hamster', 'hand_dryer', 'handbag', 'handgun', 'harbor_seal', 'harmonica', 'harp', 'harpsichord', 'headphones', 'heart_rate_monitor', 'heater', 'hedgehog', 'helicopter', 'high_heels', 'hiking_equipment', 'hippopotamus', 'hockey_puck', 'hockey_stick', 'honeycomb', 'honeydew', 'horizontal_bar', 'horse', 'hot_dog', 'house', 'house_car_key', 'houseplant', 'humidifier', 'ice_cream', 'indoor_rower', 'infant_bed', 'ipod', 'isopod', 'jacket', 'jacuzzi', 'jaguar_animal', 'jeans', 'jellyfish', 'jet_ski', 'jug', 'juice', 'juicer', 'kale', 'kangaroo', 'kettle', 'kitchen_and_dining_room_table', 'kite', 'kiwi', 'knife', 'koala', 'lacrosse_ball', 'lacrosse_stick', 'ladder', 'ladle', 'ladybug', 'lamp', 'lamp_shade', 'lantern', 'laptop', 'laptop_charger', 'lavender_plant', 'lemon', 'lemonade', 'leopard', 'lettuce', 'light_bulb', 'light_switch', 'lighthouse', 'lily', 'lime', 'limousine', 'lion', 'lipstick', 'lizard', 'lobster', 'loveseat', 'lynx', 'magpie', 'mango', 'maple', 'maracas', 'measuring_cup', 'mechanical_fan', 'microphone', 'microwave_oven', 'milk', 'miniskirt', 'mirror', 'missile', 'mixer', 'mixing_bowl', 'mobile_phone', 'monkey', 'motorcycle', 'mouse', 'mouthwash', 'muffin', 'mug', 'mule', 'mushroom', 'musical_keyboard', 'mussel', 'nail_construction', 'napkin', 'necklace', 'nectarine', 'night_light', 'nightstand', 'notebook', 'oboe', 'office_building', 'onion', 'orange', 'organ', 'ostrich', 'otter', 'oven', 'owl', 'oyster', 'paddle', 'palm_tree', 'pancake', 'panda', 'papaya', 'paper', 'paper_cutter', 'paper_towel', 'parachute', 'parking_meter', 'parrot', 'parsnip', 'passport', 'pasta', 'pasta_and_noodles', 'pattypan_squash', 'peach', 'peacock', 'pear', 'pen', 'pencil', 'pencil_case', 'pencil_sharpener', 'penguin', 'peppers', 'perfume', 'personal_flotation_device', 'phone_charger', 'piano', 'picnic_basket', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pitcher_container', 'pizza', 'pizza_cutter', 'plastic_bag', 'plate', 'platter', 'playstation', 'plum', 'polar_bear', 'police_car', 'pomegranate', 'pop_tarts', 'popcorn', 'porch', 'porcupine', 'pork', 'post_it', 'poster', 'potato', 'pottery', 'power_plugs_and_sockets', 'prawn', 'pressure_cooker', 'pretzel', 'printer', 'projector', 'pumpkin', 'punching_bag', 'rabbit', 'raccoon', 'radicchio', 'radish', 'raspberry', 'ratchet_device', 'raven', 'rays_and_skates', 'receipt', 'red_panda', 'red_tomato', 'refrigerator', 'remote_control', 'rhinoceros', 'rhubarb', 'rifle', 'ring', 'ring_binder', 'robotic_vacuum', 'rocket', 'roller_skates', 'rose', 'rugby_ball', 'ruler', 'salad', 'salmon', 'salt_and_pepper_shakers', 'sandal', 'saucer', 'sausage', 'saxophone', 'scale', 'scallop', 'scarf', 'scissors', 'scoreboard', 'scorpion', 'screwdriver', 'sea_lion', 'sea_turtle', 'seahorse', 'seat_belt', 'segway', 'semi_truck_truck_with_long_trailer', 'serving_tray', 'sewing_machine', 'shallot', 'shark', 'shaving_cream', 'sheep', 'shelf', 'shirt', 'shorts', 'shotgun', 'shower', 'shrimp', 'sink', 'skateboard', 'ski', 'skull', 'skunk', 'skyscraper', 'slow_cooker', 'snail', 'snake', 'snowboard', 'snowman', 'snowmobile', 'snowplow', 'soap', 'soap_dispenser', 'soccer_ball', 'sock', 'sofa', 'sombrero', 'sound_bar', 'sparrow', 'spatula', 'speaker_stereo_equipment', 'spice_rack', 'spider', 'spinach', 'spoon', 'sports_uniform', 'squid', 'squirrel', 'stairs', 'stapler', 'starfish', 'stationary_bicycle', 'stethoscope', 'stool', 'stop_sign', 'strawberry', 'street_light', 'stretcher', 'studio_couch', 'submarine', 'submarine_sandwich', 'suit', 'suitcase', 'sun_hat', 'sunglasses', 'surfboard', 'sushi', 'swan', 'sweet_potato', 'swim_cap', 'swimming_pool', 'swimwear', 'sword', 'syringe', 'table_tennis_racket', 'tablet_computer', 'taco', 'tangerine', 'tank', 'tap', 'tart', 'taxi', 'tea', 'tea_cup', 'teapot', 'teddy_bear', 'television', 'tennis_ball', 'tennis_racket', 'tent', 'thermostat', 'tiara', 'tick', 'tie', 'tiger', 'tin_can', 'tire', 'toaster', 'toilet', 'toilet_paper', 'tomato', 'toothbrush', 'toothpaste', 'torch', 'tortoise', 'towel', 'tower', 'traffic_light', 'train', 'training_bench', 'trampoline', 'treadmill', 'tree_house', 'tripod', 'trombone', 'truck', 'trumpet', 'turkey', 'turnip', 'umbrella', 'unicycle', 'vacuum', 'van', 'vase', 'vehicle_registration_plate', 'violin', 'volleyball_ball', 'waffle', 'waffle_iron', 'wall_clock', 'wallet', 'wardrobe', 'washing_machine', 'waste_container', 'watch', 'water_glass', 'watermelon', 'whale', 'wheel', 'wheelchair', 'whisk', 'whiteboard', 'willow', 'window', 'window_blind', 'wine', 'wine_glass', 'wine_rack', 'winter_melon', 'wok', 'wood_burning_stove', 'woodpecker', 'worm', 'wrench', 'xbox', 'yoga_mat', 'zebra', 'zucchini', )
    PALETTE = [(242, 220, 241), (125, 162, 148), (233, 150, 241), (164, 74, 186), (228, 209, 225), (46, 197, 195), (206, 206, 156), (207, 250, 167), (136, 164, 93), (62, 118, 189), (9, 9, 210), (152, 41, 191), (161, 115, 13), (5, 136, 196), (171, 44, 119), (150, 17, 153), (75, 227, 72), (62, 34, 98), (249, 155, 233), (88, 110, 138), (43, 175, 103), (116, 246, 194), (164, 128, 144), (171, 150, 117), (232, 79, 144), (126, 7, 58), (71, 120, 187), (248, 26, 170), (53, 154, 251), (160, 114, 135), (93, 186, 77), (214, 115, 179), (192, 59, 172), (1, 102, 21), (183, 76, 211), (113, 85, 171), (176, 6, 252), (187, 228, 129), (231, 14, 121), (212, 157, 111), (221, 219, 42), (87, 112, 83), (135, 155, 19), (18, 17, 114), (209, 51, 239), (36, 5, 69), (226, 249, 4), (253, 118, 132), (214, 103, 24), (179, 75, 54), (62, 131, 137), (226, 208, 121), (69, 250, 247), (91, 169, 197), (1, 179, 96), (18, 174, 219), (153, 190, 160), (10, 71, 124), (36, 46, 66), (218, 80, 117), (195, 61, 238), (178, 203, 14), (172, 184, 57), (35, 254, 230), (197, 152, 227), (186, 107, 208), (174, 31, 183), (88, 132, 63), (115, 238, 23), (169, 221, 18), (142, 39, 240), (251, 212, 182), (67, 32, 77), (208, 33, 142), (150, 24, 95), (213, 250, 62), (176, 20, 55), (170, 233, 124), (19, 106, 26), (62, 21, 75), (234, 180, 203), (61, 235, 174), (2, 157, 54), (135, 114, 100), (241, 15, 215), (43, 226, 255), (144, 201, 1), (203, 230, 61), (87, 2, 226), (23, 57, 205), (9, 148, 158), (187, 15, 69), (238, 28, 248), (209, 117, 33), (100, 181, 135), (202, 10, 73), (7, 244, 53), (79, 57, 78), (216, 215, 189), (49, 106, 29), (239, 108, 159), (20, 111, 102), (51, 244, 101), (29, 44, 65), (155, 29, 150), (177, 139, 10), (76, 48, 71), (180, 215, 12), (16, 202, 236), (215, 8, 70), (73, 184, 140), (155, 74, 197), (208, 69, 18), (14, 255, 214), (51, 199, 14), (213, 134, 251), (244, 86, 198), (92, 135, 37), (99, 117, 0), (36, 52, 116), (160, 214, 207), (149, 123, 203), (35, 54, 201), (224, 190, 138), (92, 240, 85), (12, 132, 87), (151, 89, 164), (254, 94, 66), (241, 166, 136), (81, 156, 114), (133, 203, 133), (140, 25, 177), (173, 34, 102), (197, 215, 155), (181, 165, 99), (103, 49, 171), (87, 118, 188), (145, 32, 146), (80, 173, 26), (190, 114, 207), (150, 145, 14), (162, 177, 6), (85, 55, 112), (226, 50, 109), (111, 56, 185), (94, 190, 22), (136, 184, 115), (50, 194, 43), (45, 253, 156), (95, 75, 60), (4, 210, 174), (4, 41, 122), (34, 121, 71), (145, 218, 121), (55, 129, 99), (112, 155, 57), (246, 180, 250), (30, 64, 80), (181, 34, 20), (98, 114, 60), (169, 1, 63), (245, 48, 14), (134, 187, 158), (171, 121, 18), (255, 161, 89), (194, 106, 119), (40, 36, 75), (129, 231, 188), (249, 74, 215), (129, 118, 199), (222, 233, 193), (201, 108, 39), (193, 254, 71), (92, 171, 156), (176, 29, 32), (89, 22, 177), (188, 166, 99), (62, 37, 31), (51, 202, 81), (35, 76, 173), (10, 192, 14), (193, 207, 190), (52, 219, 52), (99, 127, 180), (99, 20, 156), (206, 52, 202), (228, 174, 139), (97, 2, 89), (199, 239, 137), (238, 148, 204), (77, 135, 234), (174, 20, 129), (32, 102, 53), (58, 186, 216), (60, 197, 3), (82, 213, 197), (123, 210, 197), (172, 56, 184), (188, 147, 34), (150, 167, 25), (41, 207, 47), (205, 36, 215), (13, 162, 36), (166, 230, 23), (89, 198, 20), (147, 18, 149), (111, 142, 199), (176, 132, 110), (237, 21, 164), (237, 152, 198), (200, 187, 70), (53, 53, 164), (8, 109, 189), (93, 48, 64), (247, 206, 25), (179, 148, 0), (80, 181, 195), (25, 24, 110), (29, 56, 30), (24, 88, 51), (141, 40, 189), (10, 41, 75), (236, 152, 250), (77, 33, 219), (68, 40, 204), (89, 1, 62), (125, 144, 133), (0, 7, 216), (150, 97, 57), (45, 80, 195), (203, 113, 189), (51, 35, 253), (178, 53, 178), (81, 101, 27), (80, 95, 101), (50, 181, 165), (195, 155, 63), (19, 246, 111), (7, 156, 29), (250, 147, 185), (133, 1, 235), (42, 31, 69), (45, 248, 212), (221, 224, 50), (215, 96, 226), (193, 193, 195), (103, 165, 131), (131, 207, 102), (16, 97, 204), (230, 52, 244), (81, 37, 195), (204, 168, 130), (140, 2, 239), (220, 161, 122), (242, 39, 72), (126, 70, 42), (167, 118, 161), (40, 6, 111), (183, 83, 15), (144, 217, 179), (253, 236, 77), (232, 169, 122), (175, 237, 232), (64, 237, 50), (124, 39, 207), (139, 14, 37), (246, 18, 242), (150, 198, 53), (185, 191, 160), (253, 117, 15), (5, 31, 217), (192, 55, 166), (13, 82, 71), (173, 29, 148), (193, 49, 86), (84, 138, 219), (140, 145, 20), (191, 135, 220), (16, 183, 208), (116, 52, 147), (9, 191, 177), (62, 220, 254), (51, 147, 157), (196, 71, 214), (41, 29, 252), (12, 82, 0), (137, 123, 252), (236, 221, 149), (27, 97, 84), (10, 126, 150), (1, 231, 82), (218, 40, 163), (144, 136, 68), (57, 173, 90), (101, 217, 253), (104, 154, 193), (198, 239, 51), (212, 59, 119), (207, 64, 177), (197, 17, 110), (68, 28, 248), (154, 191, 15), (200, 5, 123), (253, 169, 22), (75, 157, 151), (60, 215, 247), (119, 82, 18), (66, 205, 115), (45, 255, 3), (166, 253, 214), (65, 243, 120), (244, 121, 6), (10, 39, 150), (179, 245, 42), (160, 100, 233), (156, 109, 19), (109, 70, 131), (46, 86, 53), (186, 94, 99), (19, 188, 128), (254, 5, 240), (186, 84, 202), (69, 32, 81), (190, 133, 227), (77, 123, 44), (246, 21, 216), (73, 59, 30), (93, 152, 68), (118, 93, 38), (67, 194, 245), (177, 96, 125), (240, 111, 158), (66, 175, 217), (34, 100, 175), (23, 62, 224), (79, 233, 105), (105, 198, 190), (194, 39, 229), (139, 203, 115), (240, 249, 102), (84, 93, 122), (191, 173, 70), (61, 153, 234), (40, 67, 254), (169, 130, 130), (21, 211, 135), (230, 160, 88), (158, 70, 198), (156, 216, 10), (150, 237, 222), (30, 190, 59), (191, 71, 77), (73, 201, 222), (110, 187, 61), (212, 79, 27), (30, 59, 209), (166, 37, 45), (211, 177, 217), (231, 129, 217), (241, 124, 193), (182, 127, 200), (110, 185, 166), (19, 102, 178), (242, 197, 39), (122, 37, 75), (192, 142, 148), (253, 20, 253), (181, 181, 37), (141, 74, 118), (116, 15, 80), (120, 132, 94), (119, 38, 77), (0, 218, 255), (77, 13, 237), (61, 196, 48), (54, 70, 81), (211, 234, 218), (102, 69, 13), (141, 35, 35), (90, 188, 65), (61, 136, 239), (81, 138, 157), (186, 72, 210), (234, 27, 93), (21, 171, 166), (135, 246, 23), (76, 74, 12), (79, 96, 117), (184, 201, 222), (210, 255, 58), (117, 58, 98), (81, 228, 216), (83, 148, 50), (172, 208, 85), (108, 79, 225), (67, 217, 190), (105, 150, 245), (15, 206, 120), (230, 137, 196), (63, 14, 20), (81, 158, 194), (138, 46, 40), (190, 46, 122), (129, 227, 162), (233, 187, 20), (252, 72, 217), (202, 119, 199), (22, 5, 178), (243, 216, 198), (88, 190, 64), (204, 98, 156), (250, 127, 9), (146, 164, 89), (101, 205, 92), (76, 210, 79), (225, 154, 57), (212, 97, 8), (35, 213, 139), (71, 51, 227), (117, 70, 164), (4, 41, 252), (108, 122, 51), (4, 146, 171), (247, 166, 175), (227, 252, 0), (32, 84, 13), (210, 180, 41), (156, 37, 72), (89, 249, 228), (84, 29, 64), (226, 174, 208), (39, 153, 77), (232, 186, 18), (14, 73, 191), (168, 58, 232), (137, 102, 217), (185, 198, 242), (226, 8, 184), (191, 46, 61), (14, 85, 133), (79, 87, 70), (168, 46, 178), (31, 195, 49), (242, 223, 190), (102, 200, 61), (3, 10, 207), (22, 223, 33), (57, 224, 139), (203, 160, 253), (91, 97, 154), (30, 178, 14), (49, 120, 84), (101, 54, 223), (183, 33, 19), (85, 39, 173), (84, 101, 236), (241, 48, 206), (240, 141, 79), (70, 168, 94), (112, 169, 70), (107, 119, 148), (201, 157, 224), (103, 232, 122), (84, 84, 10), (22, 128, 153), (64, 252, 17), (77, 17, 201), (216, 47, 18), (5, 63, 165), (202, 228, 15), (62, 32, 122), (0, 167, 157), (102, 151, 230), (69, 240, 196), (178, 111, 225), (42, 201, 47), (169, 126, 199), (242, 38, 183), (94, 140, 192), (13, 134, 227), (58, 179, 90), (174, 200, 203), (23, 167, 68), (19, 205, 120), (182, 220, 30), (142, 224, 125), (141, 40, 28), (134, 250, 26), (195, 18, 40), (74, 199, 98), (207, 82, 110), (146, 5, 103), (61, 248, 134), (198, 101, 129), (19, 247, 121), (88, 16, 221), (167, 181, 33), (112, 206, 74), (210, 23, 97), (223, 249, 76), (151, 185, 26), (11, 33, 220), (140, 130, 124), (160, 187, 220), (51, 56, 113), (76, 207, 35), (50, 46, 88), (199, 53, 11), (61, 195, 41), (60, 15, 119), (197, 226, 159), (171, 81, 136), (53, 84, 22), (68, 243, 147), (152, 18, 237), (191, 70, 156), (196, 70, 61), (35, 149, 71), (101, 236, 89), (75, 48, 89), (235, 84, 199), (173, 71, 43), (101, 105, 176), (203, 34, 77), (31, 128, 215), (14, 207, 1), (172, 79, 33), (84, 9, 62), (92, 126, 93), (76, 187, 56), (30, 43, 64), (127, 164, 207), (178, 63, 183), (204, 115, 112), (236, 169, 233), (130, 113, 22), (181, 220, 135), (211, 213, 210), (90, 13, 86), (159, 74, 19), (140, 211, 223), (161, 34, 209), (207, 154, 250), (245, 30, 79), (33, 184, 202), (103, 236, 164), (97, 118, 181), (241, 114, 148), (222, 6, 18), (128, 147, 148), (165, 42, 165), (241, 147, 121), (12, 133, 4), (77, 248, 254), (231, 87, 193), (29, 64, 149), (232, 122, 82), (55, 210, 34), (43, 234, 236), (53, 249, 35), (150, 132, 160), (247, 246, 209), (199, 33, 163), (71, 185, 6), (17, 188, 115), (12, 93, 16), (44, 95, 96), (8, 93, 255), (40, 27, 81), (195, 242, 211), (209, 159, 47), (82, 115, 142), (50, 148, 42), (172, 243, 207), (214, 156, 235), (217, 168, 159), (198, 110, 202), (93, 150, 58), (108, 119, 65), (107, 126, 40), (159, 76, 225), (42, 47, 209), (221, 155, 204), (2, 229, 32), (132, 40, 255), (118, 10, 60), (178, 215, 232), (159, 178, 221), (142, 185, 97), (56, 220, 37), (82, 178, 235), (87, 51, 162), (140, 69, 20), (16, 47, 213), (210, 205, 105), (137, 201, 34), (36, 114, 7), (210, 199, 49), (13, 223, 75), (0, 98, 129), (98, 161, 18), (44, 6, 20), (231, 243, 73), (112, 213, 15), (44, 221, 252), (228, 119, 123), (248, 85, 254), (105, 4, 132), (168, 2, 116), (128, 11, 164), (218, 112, 225), (128, 187, 15), (147, 246, 219), (149, 36, 64), (73, 223, 47), (175, 24, 14), (121, 15, 147), (142, 91, 18), (212, 62, 15), (107, 146, 232), (115, 75, 81), (225, 97, 252), (3, 157, 28), (181, 59, 111), (255, 102, 53), (49, 151, 168), (93, 218, 8), (104, 165, 199), (31, 53, 209), (45, 180, 19), (16, 35, 103), (176, 3, 41), (99, 202, 120), (110, 244, 152), (80, 33, 35), (101, 165, 87), (90, 217, 117), (214, 134, 219), ]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
