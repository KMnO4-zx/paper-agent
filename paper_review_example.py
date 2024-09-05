from src.perform_review import load_paper, perform_review
from src.prompt import reviewer_system_prompt_neg, reviewer_system_prompt_base, reviewer_system_prompt_pos
from openai import OpenAI
import json
import os

import pprint

# gpt-4o
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# deepseek-chat
deepseek_clinet = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=os.getenv('DEEPSEEK_BASE_URL'))

# load paper
text = load_paper("./example_data/attention.pdf")

"""
reviewer_system_prompt: 有三个等级
    reviewer_system_prompt_neg  : 严格模式
    reviewer_system_prompt_base : 中等模式
    reviewer_system_prompt_pos  : 宽松模式
"""
# 建议使用 gpt-4o review，使用 deepseek-chat 或许会找到一些意想不到的惊喜~
review = perform_review(text, 'gpt-4o-2024-08-06', openai_client, num_reviews_ensemble=1, num_reflections=2, reviewer_system_prompt=reviewer_system_prompt_neg)

with open('review.txt', 'a') as f:
    json.dump(review, f, indent=4)
    f.write('\n\n\n')  # 添加换行符以便于区分多个 JSON 对象

pprint.pp(review)