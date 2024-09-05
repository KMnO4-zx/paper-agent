import json
import os
import os.path as osp
import time
from typing import List, Dict, Union
from src.llm import get_response_from_llm, extract_json_between_markers
import openai
import requests
import backoff

from src.prompt import idea_first_prompt, idea_reflection_prompt, novelty_prompt, novelty_system_msg

S2_API_KEY = os.getenv("S2_API_KEY")

def generate_ideas(
    base_dir,
    client,
    model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
):
    # 如果 skip_generation 为真，则跳过生成过程并从文件中加载现有的想法
    if skip_generation:
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas  # 返回从文件中加载的想法
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")  # 文件不存在
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")  # JSON 解码错误

    # 初始化一个存储想法的列表
    idea_str_archive = []
    
    # 从文件中加载种子想法并将其转换为字符串格式
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    # 读取包含实验代码的文件内容
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    # 读取包含提示信息的文件内容
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    # 提取系统提示
    idea_system_prompt = prompt["system"]

    # 开始生成想法，最多生成 max_num_generations 次
    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            # 将之前生成的想法转化为字符串
            prev_ideas_string = "\n\n".join(idea_str_archive)

            # 消息历史初始化为空
            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            
            # 使用 LLM 生成新的想法
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            
            # 解析输出，尝试从中提取 JSON 数据
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # 如果反思次数大于1，则进行多次迭代改进
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    # 再次解析输出，尝试从中提取 JSON 数据
                    json_output = extract_json_between_markers(text)
                    assert (
                        json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    # 如果输出中包含 "I am done" 字样，则认为已收敛，提前退出循环
                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            # 将新生成的想法加入存档
            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")  # 捕获异常并打印错误信息
            continue

    # 保存生成的所有想法到文件
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas  # 返回生成的想法

def generate_next_idea(
    base_dir,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    # 初始化想法存档，并获取当前存档的大小
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    # 如果存档为空，则加载种子想法
    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:  # 仅加载第一个种子想法
            idea_archive.append(seed_idea)
    else:
        # 否则，从文件中读取实验代码和提示
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        # 尝试生成想法，最多进行 max_attempts 次尝试
        for _ in range(max_attempts):
            try:
                # 将现有想法转换为字符串形式
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                # 初始化消息历史
                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                
                # 使用 LLM 生成新的想法
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                
                # 解析输出，尝试从中提取 JSON 数据
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # 如果反思次数大于1，则进行多次迭代改进
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        
                        # 再次解析输出，尝试从中提取 JSON 数据
                        json_output = extract_json_between_markers(text)
                        assert (
                            json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        # 如果输出中包含 "I am done" 字样，则认为已收敛，提前退出循环
                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                # 将新生成的想法加入存档
                idea_archive.append(json_output)
                break  # 成功生成想法，退出尝试循环
            except Exception as e:
                print(f"Failed to generate idea: {e}")  # 捕获异常并打印错误信息
                continue

    # 保存生成的所有想法到文件
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive  # 返回更新后的想法存档

def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    # 检查查询字符串是否为空
    if not query:
        return None

    # 发起 HTTP GET 请求，查询论文数据
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    
    # 打印响应状态码
    # print(f"Response Status Code: {rsp.status_code}")
    
    # 打印响应内容的前500个字符
    # print(f"Response Content: {rsp.text[:500]}")
    
    # 如果响应状态码不是200，则引发HTTPError异常
    rsp.raise_for_status()
    
    # 解析响应的 JSON 数据
    results = rsp.json()
    
    # 获取总结果数
    total = results["total"]
    
    # 等待1秒钟，以防止频繁请求
    time.sleep(1.0)
    
    # 如果没有结果，返回 None
    if not total:
        return None
    
    # 获取论文数据列表
    papers = results["data"]
    return papers

def check_idea_novelty(
    ideas,  # 需要检查的想法列表
    base_dir,  # 存放实验代码和提示的基础目录
    client,  # 用于与LLM交互的客户端对象
    model,  # 用于执行查询的LLM模型名称
    max_num_iterations=10,  # 最大迭代次数，默认值为10
):
    # 读取实验代码文件 experiment.py
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    
    # 读取提示文件 prompt.json
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]  # 提取任务描述

    # 遍历每个想法，检查其创新性
    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")  # 如果已经检查过，跳过
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False  # 初始化标志，表示是否为创新
        msg_history = []  # 消息历史，用于跟踪对话
        papers_str = ""  # 存储找到的论文信息

        for j in range(max_num_iterations):
            try:
                # 调用 LLM 获取响应
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,  # 当前轮次
                        num_rounds=max_num_iterations,  # 总轮次
                        idea=idea,  # 当前想法
                        last_query_results=papers_str,  # 上一轮查询结果
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,  # 总轮次
                        task_description=task_description,  # 任务描述
                        code=code,  # 实验代码
                    ),
                    msg_history=msg_history,  # 消息历史
                )
                
                # 检查响应中是否包含“novel”或“not novel”的决策
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                # 解析输出中的 JSON 数据
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                # 搜索相关论文
                query = json_output["Query"]  # 从 JSON 中获取查询字符串
                papers = search_for_papers(query, result_limit=10)  # 查询论文
                if papers is None:
                    papers_str = "No papers found."

                # 将找到的论文格式化为字符串
                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel  # 将创新性结果存入想法字典中

    # 将结果保存到 JSON 文件中
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas
