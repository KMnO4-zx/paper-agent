import os
import numpy as np
import json
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
from llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)
import pprint
import openai

from prompt import *

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def load_paper(pdf_path, num_pages=None, min_size=100):
    """
    从 PDF 文件中加载文本。

    参数:
    pdf_path (str): PDF 文件的路径。
    num_pages (int, optional): 要加载的页数。如果为 None，则加载所有页。
    min_size (int, optional): 文本的最小长度。如果文本长度小于此值，则抛出异常。

    返回:
    str: 提取的文本。
    """
    try:
        # 尝试使用 pymupdf4llm 库
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        # 如果 pymupdf4llm 库失败，打印错误并尝试使用 pymupdf 库
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # 打开文档
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # 遍历文档页
                text = text + page.get_text()  # 获取 UTF-8 编码的纯文本
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            # 如果 pymupdf 库失败，打印错误并尝试使用 pypdf 库
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]

def get_meta_review(model, client, temperature, reviews):
    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
Review {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = neurips_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review

def perform_review(
    text,  # 待评审的论文文本
    model,  # 使用的语言模型
    client,  # 客户端对象，用于与语言模型通信
    num_reflections=1,  # 反思的次数，用于进一步优化评审结果
    num_reviews_ensemble=1,  # 集成评审的数量，决定生成多少个独立评审进行合并
    temperature=0.75,  # 温度参数，控制生成文本的随机性
    msg_history=None,  # 消息历史，用于跟踪生成过程中的对话历史
    return_msg_history=False,  # 是否返回消息历史
    reviewer_system_prompt=reviewer_system_prompt_neg,  # 评审生成时使用的系统提示
    review_instruction_form=neurips_form,  # 评审指令的表单
):
    
    base_prompt = review_instruction_form
    # 在提示文本中添加需要评审的论文内容
    base_prompt += f"""
Here is the paper you are asked to review:
```
{text}
```"""
    # 如果集成评审数量大于1，进行多次评审并合并结果
    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            # Higher temperature to encourage diversity.
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # Replace numerical scores with the average of the ensemble.
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))

        # Rewrite the message history with the valid one and new aggregated review.
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
THOUGHT:
I will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            # print(f"Relection: {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                # print(f"Review generation converged after {j + 2} iterations.")
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review