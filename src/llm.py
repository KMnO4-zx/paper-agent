import backoff
import openai
import json
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 使用 backoff 库装饰器来处理 API 调用中的速率限制和超时异常，自动重试
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
    msg,  # 用户输入的消息
    client,  # 用于与 API 交互的客户端对象
    model,  # 指定使用的模型
    system_message,  # 系统消息，用于设定对话上下文
    print_debug=False,  # 是否打印调试信息
    msg_history=None,  # 对话的历史记录
    temperature=0.75,  # 生成的文本的多样性
):
    if msg_history is None:
        msg_history = []  # 如果没有提供历史记录，则初始化为空列表

    # 如果模型是 OpenAI 系列的 GPT-4o 模型之一
    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4o",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]  # 将用户消息添加到历史记录中
        response = client.chat.completions.create(
            model=model,  # 使用的模型名称
            messages=[
                {"role": "system", "content": system_message},  # 系统消息
                *new_msg_history,  # 历史消息记录
            ],
            temperature=temperature,  # 生成文本的多样性
            max_tokens=3000,  # 最大生成的 token 数量
            n=1,  # 请求生成一个响应
            stop=None,  # 没有特定的停止条件
            seed=0,  # 设置随机种子，确保生成的一致性
        )
        content = response.choices[0].message.content  # 从响应中提取生成的文本内容
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]  # 更新历史记录
    elif model == "deepseek-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")


    # 如果设置了打印调试信息
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)  # 打印调试分隔符
        for j, msg in enumerate(new_msg_history):  # 遍历打印消息历史记录
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)  # 打印生成的内容
        print("*" * 21 + " LLM END " + "*" * 21)  # 打印调试分隔符
        print()

    return content, new_msg_history  # 返回生成的内容和更新后的消息历史记录


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg,  # 用户输入的消息
    client,  # 用于与 API 交互的客户端对象
    model,  # 指定使用的模型
    system_message,  # 系统消息，用于设定对话上下文
    print_debug=False,  # 是否打印调试信息
    msg_history=None,  # 对话的历史记录
    temperature=0.75,  # 生成的文本的多样性
    n_responses=1,  # 需要生成的响应数量
):
    if msg_history is None:
        msg_history = []  # 如果没有提供历史记录，则初始化为空列表

    # 如果指定的模型是 OpenAI 系列的 GPT-4o 模型之一
    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]  # 将用户消息添加到历史记录中
        response = client.chat.completions.create(
            model=model,  # 使用的模型名称
            messages=[
                {"role": "system", "content": system_message},  # 系统消息
                *new_msg_history,  # 历史消息记录
            ],
            temperature=temperature,  # 生成文本的多样性
            max_tokens=3000,  # 最大生成的 token 数量
            n=n_responses,  # 请求生成的响应数量
            stop=None,  # 没有特定的停止条件
            seed=0,  # 设置随机种子，确保生成的一致性
        )
        content = [r.message.content for r in response.choices]  # 从响应中提取生成的文本内容
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content  # 将每个响应加入新的历史记录
        ]
    elif model == "deepseek-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    # 如果模型不在支持的列表中
    else:
        raise ValueError(f"Model {model} not supported.")  # 抛出异常，模型不支持

    # 如果设置了打印调试信息
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)  # 打印调试分隔符
        for j, msg in enumerate(new_msg_history[0]):  # 遍历打印第一条消息历史记录
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)  # 打印生成的内容
        print("*" * 21 + " LLM END " + "*" * 21)  # 打印调试分隔符
        print()

    return content, new_msg_history  # 返回生成的内容和更新后的消息历史记录

def extract_json_between_markers(llm_output):
    # 定义 JSON 开始和结束的标记
    json_start_marker = "```json"
    json_end_marker = "```"

    # 找到 JSON 字符串的开始和结束索引
    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)  # 将起始索引移动到标记之后的位置
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None  # 如果没有找到开始标记，则返回 None

    if end_index == -1:
        return None  # 如果没有找到结束标记，则返回 None

    # 提取 JSON 字符串
    json_string = llm_output[start_index:end_index].strip()  # 去除前后空格
    try:
        parsed_json = json.loads(json_string)  # 尝试解析 JSON 字符串
        return parsed_json  # 如果成功，返回解析后的 JSON 对象
    except json.JSONDecodeError:
        return None  # 如果解析失败（无效的 JSON 格式），返回 None