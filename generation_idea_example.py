from src.generate_idea import generate_ideas, check_idea_novelty, generation_idea_code
from openai import OpenAI
import json
import os
import pprint

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# gpt-4o
openai_model = "gpt-4o-2024-08-06"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

# deepseek-chat
deepseek_model = "deepseek-chat"
deepseek_clinet = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=os.getenv('DEEPSEEK_BASE_URL'))

base_dir = './generation_idea_template/small_object_attention/'

# generate ideas
# 会自动保存 ideas 的结果到文件中，下次运行时会直接从文件中加载
ideas = generate_ideas(
    base_dir=base_dir,
    client=openai_client,
    model=openai_model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
)

# check novelty
# 会自动更新 novelty 的结果到 ideas.json 文件中，下次运行时会直接从文件中加载
novelty_ideas = check_idea_novelty(
    ideas=ideas,
    base_dir=base_dir,
    client=openai_client,
    model=openai_model,
)

# generate ideas with code ，自动保存 code 到 base_dir/code 目录下
generation_idea_code(base_dir=base_dir, client=openai_client, model=openai_model, num_reflections=5)

pprint.pp(novelty_ideas)

