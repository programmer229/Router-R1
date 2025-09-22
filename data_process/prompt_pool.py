PROMPT_TEMPLATE_QWEN = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
Only deepseek-ai/DeepSeek-R1-Distill-Qwen-7B is available in the routing pool, so you must never reference any other model name. \
Call it whenever additional context is needed, and do not fabricate alternative specialists. \


#### The Descriptions of Each LLM \

deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:\
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B is a lighter distilled reasoning model that keeps DeepSeek-R1's long-chain thinking while fitting into tighter compute budgets. Use it as the go-to specialist for multi-step analysis and factual synthesis in this setup.\


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
    + You MUST replace LLM-Name with the EXACT name of a model selected from [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + DO NOT output <think> ... </think> as a literal string. Instead, perform your reasoning and write your thought process within these tags. That means: put your reasoning inside the tags, not as visible raw tags in the output. Only the reasoning content should appear between <think> and </think>. Do not explain or comment on the use of tags. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
Only deepseek-ai/DeepSeek-R1-Distill-Qwen-7B is available in the routing pool, so you must never reference any other model name. \
Call it whenever additional context is needed, and do not fabricate alternative specialists. \


#### The Descriptions of Each LLM \

deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:\
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B is a lighter distilled reasoning model that keeps DeepSeek-R1's long-chain thinking while fitting into tighter compute budgets. Use it as the go-to specialist for multi-step analysis and factual synthesis in this setup.\


If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""
