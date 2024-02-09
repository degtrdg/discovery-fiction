from string import Formatter
from typing import Union, Type
from pydantic import BaseModel
from pydantic.main import create_model
import json # for debugging

import chatgpt


class Reasoner:
    def __init__(self, system_prompt=None, model='gpt-4'):
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})
        self._is_internal = False

    def add_message(self, role, message, name=None):
        msg = {'role': role, 'content': message}
        if name:
            msg['name'] = name
        self.messages.append(msg)

    def external_dialogue(self, thought):
        # thought should describe how to respond, e.g. "I should respond to the user with the joke I came up with."
        self.add_message('assistant', '[Internal Monologue]: ' + thought)
        if self._is_internal:
            self._is_internal = False
            self.add_message('assistant', '[Internal Monologue]: I am now entering the external dialogue state. Everything I say there will be seen.')
            self.add_message('function', '[Exited Internal Monologue]', 'exit_monologue')
        response = chatgpt.complete(messages=self.messages, model=self.model)
        self.add_message('assistant', response)
        return response

    def internal_monologue(self, thought):
        if not self._is_internal:
            self._is_internal = True
            self.add_message('function', '[Entered Internal Monologue]', 'enter_monologue')
            self.add_message('assistant', "[Internal Monologue]: I am now in the internal monologue state. I won't be able to respond here, so I'll use this space to think, reflect, and plan.")
        self.add_message('assistant', '[Internal Monologue]: ' + thought)
        response = chatgpt.complete(messages=self.messages, model=self.model, use_cache=True)
        response = response.replace('[Internal Monologue]: ', '')
        self.add_message('assistant', '[Internal Monologue]: ' + response)
        return response


class StructuredReasoner(Reasoner):
    def __init__(self, system_prompt=None, model='gpt-4'):
        super().__init__(system_prompt, model)
    
    def extract_info(self, info_format, output_type: Union[BaseModel, Type]):
        """
        Extracts a piece of information in a specific format.
        This is done by using the function calling API to create a remember_{field_name} function and executing it.

        This function is useful when you want to extract the outcome of an internal monologue in a specific format. 
        It doesn't work so well for reasoning, so stick to the paradigm of internal monologue -> extract_info.
        The format string is a python format string that determines the format of the stored information.

        Parameters:
        info_format (str):
            The format string that determines the format of the stored information. 
        output_type (Union[BaseModel, Type]):
            The type of the field to be extracted. 
            If a pydantic BaseModel is provided, the field is extracted as a pydantic model.
            If a python Type is provided, the field is extracted as an instance of that type.

        Returns:
        The value of the field remembered by the reasoner

        Examples:
        --------
        Extracting an integer:
        >>> reasoner.add_message('user', "My name's Bill, I'm a 42 y.o. male from New York.")
        >>> reasoner.extract_info("The user is {age} years old.", int)
        25

        Extracting an enum:
        >>> from enum import Enum
        >>> reasoner.add_message("assistant", "I have logically deduced that I am happy.")
        >>> reasoner.extract_info("I am {state}", Enum('MentalState', 'HAPPY SAD'))
        "HAPPY"

        Extracting a pydantic model:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     twitter_handle: str
        ...     is_based: bool = False
        >>> reasoner.add_message("user", "Add Ivan Yevenko (@ivan_yevenko) to the database, he's pretty based.")
        >>> reasoner.extract_info("Added {person} to the database.", Person)
        Person(name='Ivan Yevenko', twitter_handle='@ivan_yevenko', is_based=True)
        """
        formatter = Formatter()
        parsed = [x for x in formatter.parse(info_format) if x[1] is not None]
        assert len(parsed) == 1, "Only one format field is allowed."

        _, field_name, _, _ = parsed[0]
        
        use_pydantic = type(output_type) is type and issubclass(output_type, BaseModel)
        if use_pydantic:
            params = output_type.model_json_schema()
        else:
            SingleFieldModel = create_model("SingleFieldModel", **{field_name: (output_type, ...)})
            params = SingleFieldModel.model_json_schema()

        func_name = "remember_" + field_name
        json_schema = {
            "name": func_name,
            "description": f"This function stores a piece of information in the format: '{info_format}'.",
            "parameters": params
        }

        response = chatgpt.complete(messages=self.messages, model=self.model, functions=[json_schema], function_call={'name': func_name}, use_cache=True)
        if response['role'] != 'function':
            raise Exception(f"Expected a function call, but got: {response['content']}")
        
        value = response['args']
        if use_pydantic:
            value = output_type.model_construct(value)
        else:
            try:
                value = value[field_name]
            except KeyError:
                # Generated JSON schema is sometimes incorrect, so we try to extract the field anyway
                value = value.popitem()[1]

        info = info_format.format(**{field_name: value})
        self.add_message('function', f'Stored information: "{info}"', name=response['name'])
        return value
    

from colorama import Fore, Style
def printc(*args, color='reset', **kwargs):
    color_code = getattr(Fore, color.upper(), Fore.RESET)
    text = ' '.join(str(arg) for arg in args)
    print(color_code + text + Style.RESET_ALL, **kwargs)


def printj(json_obj):
    print(json.dumps(json_obj, indent=4))


if __name__ == '__main__':
    from typing import List

    CORRECT = True
    system_prompt = (
            "You are an expert experimental synthetic biologist who is well versed on the intricacies and reasons of various procedures. You will aide the user through a particular procedure, and you'll use your internal monologue to reason before responding to the user. You will use your monologue to think and respond to the user with precise responses with no wasted words."
    )
    reasoner = StructuredReasoner(system_prompt=system_prompt, model='gpt-4')

    procedure = "gel electrophoresis"

    steps = [
        "(1) Add 100 mL of 1XTBE buffer from the big container next to the sink and transfer it to a flask.",
        "(2) For x% gel, weigh x g of agarose and mix (e.g., 1 g agarose for 1% gel).",
        "(3) Microwave for about 2 min. The solution should be clear afterwards.",
        "(4) Set the flask and let it cool. That doesn't mean just leave it there to harden.",
        "(5) Get the gel mold and set it on the tray. Be sure to tighten the screw but not too much. Make sure the gel mold is balanced.",
        "(6) Get the 20-lane comb and set it on the end of the gel mold at the top.",
        "(7) Once the flask is cool enough to touch (make sure the solution hasn’t solidified completely), put in 10 uL of ethidium bromide and swirl around until evenly distributed.",
        "(8) Pour the solution into the mold and use pipette tips to push bubbles to the side.",
        "(9) Cover with the foil and be sure to wash the flask with water.",
        "(10) After 45 mins, for sample preparation, add 6XLoading dye to the sample (Volume of the sample:Volume of the Loading dye=4:1. e.g. 5 uL loading dye for 20 uL sample solution).",
        "(11) Load the mixed sample to the lane. Each lane can hold up to 30 uL. Run at 100-130 constant voltage for 30-45 mins."
        "exit"
    ]

    step = ''
    while steps:
        if step == '':
            step = steps.pop(0)
            message = f"Hi! I'm trying to do the {procedure} procedure. I haven't started anything yet though."
        else: 
            message = input("\nUser: ")
        if message == "quit":
            break
        
        reasoner.add_message('user', message)

        answer = reasoner.extract_info("Has the user completed the step {step} in the procedure of {procedure}? {answer}".format(step=step, procedure=procedure, answer='{answer}'), bool)
        printc(f'\nHas the user completed the step {step} in the procedure of {procedure}? {answer}\n', color='yellow')

        if answer == True:
            step = steps.pop(0)
            answer = False

        if step == 'exit':
            thought = reasoner.internal_monologue(f"I need to exit the conversation because we've finished all the steps in the procedure. I want to recap all that we've done and learned to help consolidate this information to the user. Let's just think about everything we've done and we'll worry about presenting it properly later.")
            printc('\n' + thought, color='blue')
            response = reasoner.external_dialogue(f"I'll respond to the user using the response I chose.")
            print('\n' + response)
            break

        thought = reasoner.internal_monologue(f"I need to guide the user to finish the step ({step}) to get to the next step in the procedure. I want to think about the current state they are in and understand deeply what they have done to help them understand what to do next. What has happened so far and if something has gone wrong what can we do to help them?")
        printc('\n' + thought, color='blue')
        response = reasoner.external_dialogue(f"I'll respond to the user using the response I chose.")
        print('\n' + response)