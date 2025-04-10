import random

def get_random_element(list):
    random_element = list[random.randint(0, len(list) - 1)]
    return random_element

class random_character_prompts:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
                "character_type": (["girl", "boy"],),
                "hair_prompt": ("STRING", {"multiline": True,}),
                "randomize_hair": ("BOOLEAN", {"default": False}),
                "face_prompt": ("STRING", {"multiline": True,}),
                "randomize_face": ("BOOLEAN", {"default": False}),
                "torso_prompt": ("STRING", {"multiline": True,}),
                "randomize_torso": ("BOOLEAN", {"default": False}),
                "hands_prompt": ("STRING", {"multiline": True,}),
                "randomize_hands": ("BOOLEAN", {"default": False}),
                "legs_prompt": ("STRING", {"multiline": True,}),
                "randomize_legs": ("BOOLEAN", {"default": False}),
                "feet_prompt": ("STRING", {"multiline": True,}),
                "randomize_feet": ("BOOLEAN", {"default": False}),
                "body_prompt": ("STRING", {"multiline": True, }),
                "randomize_body": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("HAIR", "FACE", "TORSO", "HANDS", "LEGS", "FEET", "BODY")
    #OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, character_type, seed, hair_prompt, randomize_hair, face_prompt, randomize_face, torso_prompt, randomize_torso, hands_prompt, randomize_hands, legs_prompt, randomize_legs, feet_prompt, randomize_feet, body_prompt, randomize_body, unique_id=None, extra_pnginfo=None):
        random.seed(seed)


        colors = ["blue ", "yellow ", "red ", "green ", "purple ", "black ", "white ", "silver ", "golden ", "brown ", "ruby ", "gray ", "orange ", "pink "]
        materials = ["denim ", "leather ", "velvet ", "rugged ", "fur ", "plastic ", "metal ", "gold ", "", "gel ", "", "", ""]
        #girl_shirts = ["hoodie ", "shirt ", "top ", "dress ", "jacket ", "cardigan ", "blouse ", "sweater ", "tunic ", "t-shirt ", "camisole ", "polo ", "shirt dress ", "dress ", "blazer ", "raincoat ", "coat "]
        #girl_pants = ["shorts ", "long skirt ", "skirt ", "baggy pants ", "leggings ", "pants ", "thigh highs ", "stockings ", "pants ", "dress ", "armor "]
        hair_length = ["long ", "short ", "very long ", "very short ", "medium ", "", "", "", "", ""]
        #hair_type = ["curly ", "straight ", "messy ", "tidy ", "wavy ", "braided ", "twintails ", "ponytail ", ""]
        shoes = ["sandals ", "sneakers ", "shoes ", "boots ", "loafers ", "flip-flops ", "heels ", "cowboy boots ", "slippers "]
        hand_acc = ["fingerless gloves ", "watch ", "bracelet ", "boxing gloves ", "watch ", "bracelet ", "gloves ", "ring " "bare hands ", "hands "]
        #eye_types = ["tsundere ", "yandere ", "kind ", "soft ", "sharp ", "round, ", "cat ", "", "happy ", "motherly ", "pretty "]
        expressions = ["smiling ", "curious ", "shy ", "inquisitive ", "slightly angry ", "neutral expression ", "scared ", "sad ", "happy ", "innocent "]

        if character_type == "girl":
            if randomize_hair:
                hair_type = ["curly ", "straight ", "messy ", "tidy ", "wavy ", "braided ", "twintails ", "ponytail ", ""]
                hair = get_random_element(colors) + get_random_element(hair_length) + get_random_element(hair_type) + "hair"
            else:
                hair = hair_prompt

            if randomize_face:
                eye_types = ["tsundere ", "yandere ", "kind ", "soft ", "sharp ", "round, ", "cat ", "happy ",
                             "motherly ", "pretty "]
                face = get_random_element(colors) + get_random_element(eye_types) + "eyes, " + get_random_element(expressions)
            else:
                face = face_prompt

            if randomize_torso:
                girl_shirts = ["hoodie ", "shirt ", "top ", "dress ", "jacket ", "cardigan ", "blouse ", "sweater ","tunic ", "t-shirt ", "camisole ","polo ", "shirt dress ", "dress ", "blazer ", "raincoat ", "coat "]
                torso = get_random_element(colors) + get_random_element(materials) + get_random_element(girl_shirts)
            else:
                torso = torso_prompt

            if randomize_hands:
                hands = get_random_element(colors) + get_random_element(materials) + get_random_element(hand_acc)
            else:
                hands = hands_prompt

            if randomize_legs:
                girl_pants = ["shorts ", "long skirt ", "skirt ", "baggy pants ", "leggings ", "pants ", "thigh highs ", "stockings ", "pants ", "dress ", "armor "]
                legs = get_random_element(colors) + get_random_element(materials) + get_random_element(girl_pants)
            else:
                legs = legs_prompt

            if randomize_feet:
                feet = get_random_element(colors) + get_random_element(materials) + get_random_element(shoes)
            else:
                feet = feet_prompt

            if randomize_body:
                girl_body = ["fat ", "muscular ", "thick ", "skinny ", "toned ", "fit ", "chubby ", "normal ", "normal ", "normal " ]
                body = get_random_element(girl_body) + "body"
            else:
                body = body_prompt


            return (hair, face, torso, hands, legs, feet, body)
            #return {"ui": {"hair_prompt": hair}, "result": (hair, face, torso, hands, legs, feet)}

        elif character_type == "boy":
            if randomize_hair:
                hair_type = ["curly ", "straight ", "messy ", "wavy ", "braided ", "buzz cut ", "shaved ", "dreadlocks ", "mohawk ", "fade ", "slick back ", ""]
                hair = get_random_element(colors) + get_random_element(hair_length) + get_random_element(hair_type) + "hair"
            else:
                hair = hair_prompt

            if randomize_face:
                eye_types = ["kind ", "soft ", "sharp ", "round, ", "cat ", "happy ",
                             "motherly ", "handsome ", "shonen "]
                face = get_random_element(colors) + get_random_element(eye_types) + "eyes, " + get_random_element(expressions)
            else:
                face = face_prompt

            if randomize_torso:
                boy_shirts = ["hoodie ", "shirt ", "top ", "jacket ", "cardigan ", "sweater ","tunic ", "t-shirt ","polo ", "blazer ", "raincoat ", "coat ", "armor "]
                torso = get_random_element(colors) + get_random_element(materials) + get_random_element(boy_shirts)
            else:
                torso = torso_prompt

            if randomize_hands:
                hands = get_random_element(colors) + get_random_element(materials) + get_random_element(hand_acc)
            else:
                hands = hands_prompt

            if randomize_legs:
                boy_pants = ["shorts ", "baggy pants ", "jeans ", "pants ", "pants ", "long shorts ", "armor "]
                legs = get_random_element(colors) + get_random_element(materials) + get_random_element(boy_pants)
            else:
                legs = legs_prompt

            if randomize_feet:
                feet = get_random_element(colors) + get_random_element(materials) + get_random_element(shoes)
            else:
                feet = feet_prompt

            if randomize_body:
                boy_body = ["fat ", "muscular ", "twink ", "skinny ", "toned ", "fit ", "chubby ", "normal ", "normal ", "normal " ]
                body = get_random_element(boy_body) + "body"
            else:
                body = body_prompt

            return (hair, face, torso, hands, legs, feet, body)


class string_list_to_prompt_schedule:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt" : ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("String", )
    #OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True
    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, prompt):
        combine_strings = prompt
        prompt_schedule = ''
        
        #for y in prompt:
        #    combine_strings.append(y[0])
        
        for n in combine_strings:
            if combine_strings.index(n)<(len(combine_strings) - 1):
                prompt_schedule += '"' + str(combine_strings.index(n)) + '": "' + n + '",\n'
            else:
                prompt_schedule += '"' + str(combine_strings.index(n)) + '": "' + n + '"'
        
        return (prompt_schedule,)


class JoinStringLists:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"string_list1" : ("STRING", {"forceInput": True}),
                         "string_list2": ("STRING", {"forceInput": True}),
                         },
        }

    RETURN_TYPES = ("STRING",)
    INPUT_IS_LIST = (True, True)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "AIR Nodes"

    def doit(self, string_list1, string_list2):
        values = string_list1 + string_list2

        return (values,)


class CreateFilenameList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images" : ("IMAGE",),
                         "key_frame" : ("BOOLEAN", {"default": False}),
                         "pre_text" : ("STRING", {"multiline": False,}),
                         "app_text": ("STRING", {"multiline": False,}),
                         "file_type": (["jpeg", "png", "webp"],),
                         },
        }

    RETURN_TYPES = ("STRING",)
    #INPUT_IS_LIST = (True, True)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "AIR Nodes"

    def doit(self, images, key_frame, pre_text, app_text, file_type):

        if file_type == "jpeg":
            f_type = '.jpg'
        elif file_type == "png":
            f_type = '.png'
        elif file_type == "webp":
            f_type = '.webp'

        values = []
        
        for x in range(len((images))):

            name = ''
            if key_frame:
                name += pre_text + '_' + app_text + f_type
            else:
                name += pre_text + '-' + str("{:02d}".format(x+1)) + '_' + app_text + f_type

            values.append(name)

        return (values,)

NODE_CLASS_MAPPINGS = {
    "string_list_to_prompt_schedule": string_list_to_prompt_schedule,
    "RandomCharacterPrompts": random_character_prompts,
    "JoinStringLists": JoinStringLists,
    "CreateFilenameList": CreateFilenameList,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "string_list_to_prompt_schedule": "String List To Prompt Schedule",
    "RandomCharacterPrompts": "Random Character Prompts",
    "JoinStringLists": "Join String Lists",
    "CreateFilenameList": "Create Filename List",
    }
