# Paper-Agent

> *该项目来源于 [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) 项目，Paper: [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)*

AI-Scientist 项目很棒，实验思路设计很精巧，很值得学习。但现有模型在写论文和修改代码做实验方面还有很多不足，所以在仔细研读了 AI-Scientist 的论文和项目代码之后，将 `generation ideas` 和 `paper revierw` 部分的代码拿来出来，做了一些简单的修改，并给代码添加了详细的中文注释，方便学习。

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

### Paper Review

运行代码可以参考 `paper_review_example.py` 文件，也可以直接运行该文件。

```bash
python paper_review_example.py
```

> note: 代码的运行结果还会保存在根目录下的 `reviews.txt` 文件中。

## Blog

*后续有时间会有一个关于 AI-Scientist Blog （如果有时间的话）*

