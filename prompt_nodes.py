import random

class random_character_prompts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_type": (["girl", "boy"],),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("HAIR", "FACE", "TORSO", "HANDS", "LEGS", "FEET", "seed")

    FUNCTION = "run"

    CATEGORY = "AIR Nodes"

    def run(self, character_type, seed):
        random.seed(seed)

        colors = ["blue ", "yellow ", "red ", "green ", "purple ", "black ", "white ", "silver ", "golden ", "brown ", "hazel ", "ruby ", "gray ", "orange ", "pink "]
        materials = ["denim ", "leather ", "velvet ", "rugged ", "fur ", "polyester ", "metal ", "gold ", "", "gel ", "", "", ""]
        #girl_shirts = ["hoodie ", "shirt ", "top ", "dress ", "jacket ", "cardigan ", "blouse ", "sweater ", "tunic ", "t-shirt ", "camisole ", "polo ", "shirt dress ", "dress ", "blazer ", "raincoat ", "coat "]
        #girl_pants = ["shorts ", "long skirt ", "skirt ", "baggy pants ", "leggings ", "pants ", "thigh highs ", "stockings ", "pants ", "dress ", "armor "]
        hair_length = ["long ", "short ", "neck length ", "very long ", "very short ", "medium ", ""]
        hair_type = ["curly ", "straight ", "messy ", "tidy ", "wavy ", "braided ", "twintails ", "ponytail ", ""]
        shoes = ["sandals ", "sneakers ", "shoes ", "boots ", "loafers ", "flip-flops ", "heels ", "cowboy boots ", "slippers "]
        hand_acc = ["fingerless gloves ", "watch ", "bracelet ", "boxing gloves ", "watch ", "bracelet ", "gloves ", "ring "]
        eye_types = ["tsundere ", "yandere ", "kind ", "soft ", "sharp ", "round, ", "cat ", "frog ", "happy ", "motherly ", "pretty "]
        expressions = ["smiling ", "curious ", "shy ", "inquisitive ", "slightly angry ", "neutral expression ", "scared ", "sad ", "happy ", "innocent "]

        if character_type == "girl":

            girl_shirts = ["hoodie ", "shirt ", "top ", "dress ", "jacket ", "cardigan ", "blouse ", "sweater ", "tunic ", "t-shirt ", "camisole ",
                           "polo ", "shirt dress ", "dress ", "blazer ", "raincoat ", "coat "]
            girl_pants = ["shorts ", "long skirt ", "skirt ", "baggy pants ", "leggings ", "pants ", "thigh highs ", "stockings ", "pants ", "dress ", "armor "]

            hair = colors[random.randint(0, len(colors) - 1)] + hair_length[random.randint(0, len(hair_length) - 1)] + hair_type[random.randint(0, len(hair_type) - 1)] + "hair"
            face = colors[random.randint(0, len(colors)-1)] + eye_types[random.randint(0, len(eye_types)-1)] + "eyes, " + expressions[random.randint(0, len(expressions)-1)]
            hands = colors[random.randint(0, len(colors)-1)] + materials[random.randint(0, len(materials)-1)] + hand_acc[random.randint(0, len(hand_acc)-1)]
            feet = colors[random.randint(0, len(colors)-1)] + materials[random.randint(0, len(materials)-1)] + shoes[random.randint(0, len(shoes)-1)]
            legs = colors[random.randint(0, len(colors)-1)] + materials[random.randint(0, len(materials)-1)] + girl_pants[random.randint(0, len(girl_pants)-1)]
            torso = colors[random.randint(0, len(colors)-1)] + materials[random.randint(0, len(materials)-1)] + girl_shirts[random.randint(0, len(girl_shirts)-1)]

            return (hair, face, torso, hands, legs, feet, seed)

        elif character_type == "boy":
            hair = ""
            face = ""
            torso = ""
            hands = ""
            legs = ""
            feet = ""

            return (hair, face, torso, hands, legs, feet, seed)



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
