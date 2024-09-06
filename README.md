# Paper-Agent

> *该项目来源于 [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) 项目，Paper: [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)*

AI-Scientist 项目很棒，实验思路设计很精巧，很值得学习。但现有模型在写论文和修改代码做实验方面还有很多不足，所以在仔细研读了 AI-Scientist 的论文和项目代码之后，将 `generation ideas` 和 `paper revierw` 部分的代码拿来出来，做了一些简单的修改，并给代码添加了详细的中文注释，方便学习。

每篇论文的诞生都经过了漫长的思考和实验，目前的LLM应该还无法胜任做实验这个任务，但它可以为我们提供一些更为大胆的 idea ，以及对论文的 review。review 可以找出论文中意想不到优缺点（优点：更好的讲故事；缺点：避重就轻，扬长避短，省略不要！哈哈）。LLM generation idea 和 paper review 希望可以把你变成 ***论文打印机！*** 

> *项目所有的 prompt 可以在 `src/prompt.py` 中找到。*

## Usage

### .env

首先需要配置一些环境变量，可以参考 `.env.example` 文件。

```
OPENAI_API_KEY=''
# OPENAI_BASE_URL='' 

DEEPSEEK_API_KEY=''
DEEPSEEK_BASE_URL='https://api.deepseek.com'

# 用于S2的API Key 
S2_API_KEY=''
```

> *S2 的 API Key 可以在 [Semantic Scholar](https://www.semanticscholar.org/) 上申请。*

### Generation Ideas

首先需要准备几个文件：`experiment.py`, `prompt.json` 和 `seed_ideas.json`。

- `experiment.py` 是你的实验/模块代码，必须有。
- `prompt.json` 是对任务的描述，必须有。
- `seed_ideas.json` 是你的实验代码的种子想法，可以没有（但文件要存在）。

> 可以参考 `generation_idea_template` 文件夹中的示例。

运行代码可以参考 `generation_idea_example.py` 文件，也可以直接运行该文件。

```bash
python generation_idea_example.py
```

> note: 代码的运行结果还会保存在 `base_dir` 目录下的 `ideas.json` 文件中。

#### 代码详解

`generation_idea_example.py` 文件中大致代码如下：

- `generate_ideas`：用于生成 idea。
- `check_idea_novelty`：用于检查 idea 的新颖性。但需要申请 emantic Scholar 的 API Key。
- `generation_idea_code`：可以根据以上两个函数生成idea来生成代码。如果已经 `check novelty`，会直接生成 `novel=True` 的代码，如果没有 `check novelty`，会生成所有的代码。

```python
# gpt-4o
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
```

### Paper Review

运行代码可以参考 `paper_review_example.py` 文件，也可以直接运行该文件。

```bash
python paper_review_example.py
```

> note: 代码的运行结果还会保存在根目录下的 `reviews.txt` 文件中。

## Blog

*后续有时间会有一个关于 AI-Scientist Blog （如果有时间的话）*

