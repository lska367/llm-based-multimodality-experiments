# multimodality-rerank

多模态文档重排序系统，基于MMDocIR基准测试的实现，用于评估和比较不同多模态检索模型在长文档检索任务中的性能。

## 项目概述

multimodality-rerank是一个专注于多模态文档检索的项目，它利用MMDocIR基准测试来评估各种检索模型的性能。该项目支持页面级和布局级检索任务，并提供了多种文本和视觉检索模型的实现和比较框架。

### 主要特性

- 支持多种文本检索模型：BGE、E5、GTE、Contriever、DPR、ColBERT
- 支持多种视觉检索模型：ColPali、ColQwen、DSE-docmatix、DSE-wikiss
- 支持多种输入模态：vlm_text、ocr_text、image_binary、image_hybrid
- 提供灵活的编码和搜索接口
- 支持页面级和布局级检索任务评估

## 目录结构

```
├── .gitignore
├── .python-version
├── MMDocIR/           # MMDocIR基准测试实现
│   ├── README.md      # MMDocIR详细文档
│   ├── encode.py      # 编码脚本
│   ├── main.py        # 示例入口
│   ├── metric_eval.py # 指标评估
│   ├── scripts/       # 辅助脚本
│   ├── search.py      # 搜索脚本
│   ├── static/        # 静态资源
│   ├── text_wrapper.py # 文本模型包装器
│   └── vision_wrapper.py # 视觉模型包装器
├── README.md          # 项目说明文档
├── main.py            # 项目入口
├── pyproject.toml     # 项目配置和依赖
└── uv.lock            # 依赖锁定文件
```

## 安装指南

### 前置要求

- Python 3.13+
- uv 包管理器

### 安装步骤

1. 克隆项目仓库

```bash
git clone <repository-url>
cd LLM4Rank/experiments/multimodality_rerank
```

2. 使用uv安装依赖

```bash
uv install
```

3. 下载MMDocIR数据集和模型（可选）

```bash
cd MMDocIR
bash scripts/download_dataset.sh
bash scripts/download_model.sh
```

## 使用方法

### 基本用法

运行主程序：

```bash
python main.py
```

### 编码文档和查询

使用`encode.py`脚本对查询、页面和布局进行编码：

```bash
cd MMDocIR
python encode.py BGE --bs 256 --mode vlm_text --encode query,page,layout
```

参数说明：
- `model`：模型名称，如"BGE"、"ColPali"等
- `--bs`：批次大小
- `--mode`：输入模式，可选值：'vlm_text', 'oct_text', 'image_binary', 'image_hybrid'
- `--encode`：需要编码的对象，可选值：'query', 'page', 'layout'，用逗号分隔

### 搜索和评估

使用`search.py`脚本进行搜索和评估：

```bash
python search.py BGE --encode page,layout --encode_path encode
```

参数说明：
- `model`：模型名称
- `--encode`：需要评估的对象，可选值：'page', 'layout'，用逗号分隔
- `--encode_path`：编码结果存储路径

## 支持的模型

### 文本检索模型
- BGE：基于sentence-transformers的文本编码器
- E5：基于sentence-transformers的文本编码器
- GTE：基于sentence-transformers的文本编码器
- Contriever：基于sentence-transformers的文本编码器
- DPR：双编码器检索模型
- ColBERT：基于colbert-ai的上下文学习检索模型

### 视觉检索模型
- ColPali：基于视觉语言模型的检索适配器
- ColQwen：基于Qwen视觉语言模型的检索适配器
- DSE-docmatix：基于DSE和docmatix的检索模型
- DSE-wikiss：基于DSE和wikiss的检索模型

## 数据集

项目使用MMDocIR数据集，包含：
- 1,685个专家标注的问题
- 173,843个自举标签的问题
- 支持页面级和布局级检索任务

## 评估指标

项目支持以下评估指标：
- 页面级检索的Top-K召回率
- 布局级检索的Top-K召回率

## 依赖项

主要依赖项：
- colbert-ai>=0.2.22
- pandas>=2.3.3
- sentence-transformers>=5.1.2
- torch>=2.9.1
- transformers>=4.57.3

## 开发指南

### 添加新的检索模型

1. 在`text_wrapper.py`或`vision_wrapper.py`中实现模型包装器
2. 在`encode.py`的`get_retriever`函数中添加模型支持
3. 确保模型实现`embed_queries`和`embed_quotes`方法

### 扩展评估指标

在`metric_eval.py`中添加新的评估指标和计算逻辑。

## 参考文献

```
@misc{dong2025mmdocirbenchmarkingmultimodalretrieval,
      title={MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents}, 
      author={Kuicai Dong and Yujing Chang and Xin Deik Goh and Dexun Li and Ruiming Tang and Yong Liu},
      year={2025},
      eprint={2501.08828},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2501.08828}, 
}
```

## 许可证

- 代码：Apache 2.0
- 数据：CC BY NC 4.0

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目地址：<repository-url>
- 作者：<author-name>
- 邮箱：<author-email>

## 致谢

感谢MMDocIR团队提供的基准测试和数据集，以及所有贡献代码和建议的开发者。
