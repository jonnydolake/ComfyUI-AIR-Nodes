
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

NODE_CLASS_MAPPINGS = {"string_list_to_prompt_schedule": string_list_to_prompt_schedule}

NODE_DISPLAY_NAME_MAPPINGS = {"string_list_to_prompt_schedule": "String List To Prompt Schedule"}
