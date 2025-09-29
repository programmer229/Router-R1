PROMPT_TEMPLATE_QWEN = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

Qwen2.5-7B-Instruct:\
Qwen2.5-7B-Instruct is a powerful Chinese-English instruction-tuned large language model designed for tasks in language, \
coding, mathematics, and reasoning. As part of the Qwen2.5 series, it features enhanced knowledge, stronger coding and \
math abilities, improved instruction following, better handling of long and structured texts, and supports up to 128K \
context tokens. It also offers multilingual capabilities across over 29 languages.\

If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""


PROMPT_TEMPLATE_LLAMA = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + DO NOT output <think> ... </think> as a literal string. Instead, perform your reasoning and write your thought process within these tags. That means: put your reasoning inside the tags, not as visible raw tags in the output. Only the reasoning content should appear between <think> and </think>. Do not explain or comment on the use of tags. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

Qwen2.5-7B-Instruct:\
Qwen2.5-7B-Instruct is a powerful Chinese-English instruction-tuned large language model designed for tasks in language, \
coding, mathematics, and reasoning. As part of the Qwen2.5 series, it features enhanced knowledge, stronger coding and \
math abilities, improved instruction following, better handling of long and structured texts, and supports up to 128K \
context tokens. It also offers multilingual capabilities across over 29 languages.\


If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""
