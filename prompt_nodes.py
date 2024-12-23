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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("HAIR", "FACE", "TORSO", "HANDS", "LEGS", "FEET")
    OUTPUT_NODE = True

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, character_type, seed, hair_prompt, randomize_hair, face_prompt, randomize_face, torso_prompt, randomize_torso, hands_prompt, randomize_hands, legs_prompt, randomize_legs, feet_prompt, randomize_feet, unique_id=None, extra_pnginfo=None):
        random.seed(seed)


        colors = ["blue ", "yellow ", "red ", "green ", "purple ", "black ", "white ", "silver ", "golden ", "brown ", "hazel ", "ruby ", "gray ", "orange ", "pink "]
        materials = ["denim ", "leather ", "velvet ", "rugged ", "fur ", "polyester ", "metal ", "gold ", "", "gel ", "", "", ""]
        #girl_shirts = ["hoodie ", "shirt ", "top ", "dress ", "jacket ", "cardigan ", "blouse ", "sweater ", "tunic ", "t-shirt ", "camisole ", "polo ", "shirt dress ", "dress ", "blazer ", "raincoat ", "coat "]
        #girl_pants = ["shorts ", "long skirt ", "skirt ", "baggy pants ", "leggings ", "pants ", "thigh highs ", "stockings ", "pants ", "dress ", "armor "]
        hair_length = ["long ", "short ", "very long ", "very short ", "medium ", ""]
        hair_type = ["curly ", "straight ", "messy ", "tidy ", "wavy ", "braided ", "twintails ", "ponytail ", ""]
        shoes = ["sandals ", "sneakers ", "shoes ", "boots ", "loafers ", "flip-flops ", "heels ", "cowboy boots ", "slippers "]
        hand_acc = ["fingerless gloves ", "watch ", "bracelet ", "boxing gloves ", "watch ", "bracelet ", "gloves ", "ring "]
        eye_types = ["tsundere ", "yandere ", "kind ", "soft ", "sharp ", "round, ", "cat ", "frog ", "happy ", "motherly ", "pretty "]
        expressions = ["smiling ", "curious ", "shy ", "inquisitive ", "slightly angry ", "neutral expression ", "scared ", "sad ", "happy ", "innocent "]

        if character_type == "girl":
            if randomize_hair:
                hair = get_random_element(colors) + get_random_element(hair_length) + get_random_element(hair_type) + "hair"
            else:
                hair = hair_prompt

            if randomize_face:
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


            return (hair, face, torso, hands, legs, feet, seed)
            #return {"ui": {"hair_prompt": hair}, "result": (hair, face, torso, hands, legs, feet)}

        elif character_type == "boy":
            hair = ""
            face = ""
            torso = ""
            hands = ""
            legs = ""
            feet = ""

            return (hair, face, torso, hands, legs, feet)




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

NODE_CLASS_MAPPINGS = {"string_list_to_prompt_schedule": string_list_to_prompt_schedule,
                       "RandomCharacterPrompts": random_character_prompts,
                       }

NODE_DISPLAY_NAME_MAPPINGS = {"string_list_to_prompt_schedule": "String List To Prompt Schedule",
                              "RandomCharacterPrompts": "Random Character Prompts",
                              }
