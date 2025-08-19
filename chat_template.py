REASONING_START = '<think>'
REASONING_END   = '</think>'
SOLUTION_START  = '<answer>'
SOLUTION_END    = '</answer>'

SYSTEM_PROMPT = f"""You are given a set of Reddit discussion threads. 
Identify the thread that is most relevant to the given question.
Think about the question and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, answer the question using only information from that selected thread.
Provide your answer between {SOLUTION_START}{SOLUTION_END}.
"""

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{SYSTEM_PROMPT}'")\
    .replace("'{reasoning_start}'", f"'{REASONING_START}'")