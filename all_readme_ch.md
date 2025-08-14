# U2-Bench
基于VLMEvalKit构建的多模态评估。

## 部分模型部署说明
| 脚本文件 | 适用模型 | 说明 |
|---------|---------|------|
| `batch_test.py` | QwenVL7B、QwenVL32B、QwenVL72B | 用于测试本地部署的通义千问视觉系列模型 |
| `batch_test_xiaohu.py` | GPT系列、Claude系列、Qwen max | 用于测试通过小虎API代理访问的OpenAI和Anthropic模型以及通义千问max |
| `batch_test_volc.py` | doubao系列 | 用于测试通过火山引擎方舟平台访问的豆包模型 |
| `batch_test_xiaohu_gemini25.py` | gemini-2.5系列 | 专门优化用于测试Google Gemini 2.5模型 |
| `batch_test_gemini_fixed.py` | gemini-2.0系列 | 专门优化用于测试Google Gemini 2.0模型 |
| `batch_test_dashscope.py` | qwen2.5-vl-3b | 用于通过阿里云百炼平台测试qwen2.5-vl-3b模型 |
**以上脚本文件均在vlmeval/silicon_test中**

### 通用功能

所有脚本都支持以下基本功能：

1. 批量测试：对多个数据集和任务进行批量测试
2. 多模态输入：支持图像和文本的组合输入
3. 结果保存：自动保存测试结果到指定目录
4. 异步处理：支持异步API调用，提高测试效率

### 使用方法

#### 1. 基本用法

每个脚本都支持以下基本命令行参数：

```bash
python <脚本名>.py --task_type <任务类型> --dataset <数据集名称> --output_dir <输出目录>
```

其中：
- `<任务类型>`: 可选值为 `cla`(分类), `measurement`(测量), `seg`(分割), `report`(报告生成)
- `<数据集名称>`: 对应任务类型下的数据集编号，如 `03`, `10` 等
- `<输出目录>`: 结果保存目录，默认为 `outputs/dolphin-output`

#### 2. 特定模型测试示例

##### 测试QwenVL系列模型

```bash
# 测试QwenVL7B模型在分类任务03数据集上的表现
python batch_test.py --task_type cla --dataset 03 --model Qwen2.5-VL-7B-Instruct-Pro
```

##### 测试GPT-4o模型

```bash
# 测试GPT-4o模型在分割任务04数据集上的表现
python batch_test_xiaohu.py --task_type seg --dataset 04 --model gpt-4o-2024-08-06
```

##### 测试豆包模型

```bash
# 测试豆包模型在报告生成任务10数据集上的表现
python batch_test_volc.py --task_type report --dataset 10 --model doubao-1.5-vision-pro-32k-250115
```

##### 测试Gemini 2.5模型

```bash
# 测试Gemini 2.5模型在测量任务27数据集上的表现
python batch_test_xiaohu_gemini25.py --task_type measurement --dataset 27
```

#### 3. 测试所有数据集

使用`--test_all_datasets`参数可以测试指定任务类型下的所有数据集：

```bash
python batch_test.py --task_type cla --test_all_datasets --model Qwen2.5-VL-7B-Instruct-Pro
```

#### 4. 多任务并行测试

部分脚本支持多任务并行测试，可同时测试多个任务类型和数据集：

```bash
python batch_test.py --multi_task --task_types cla seg --datasets 03 04
```

### 输出结果

测试结果将保存在指定的输出目录中，按以下结构组织：

```
outputs/dolphin-output/
├── <任务类型>/
│   ├── <数据集>/
│   │   ├── <模型名>/
│   │   │   ├── multimodal_test_results_<时间戳>.jsonl  # 原始测试结果
│   │   │   └── <模型名>_combined_results.jsonl        # 合并后的结果
```

### 注意事项

1. 确保已正确设置相应API密钥或模型访问凭证
2. 对于本地模型测试，确保模型已正确加载并可访问
3. 网络连接问题可能导致API调用失败，脚本会自动重试
4. 对于大型数据集，可使用`--sample_limit`参数限制测试样本数量
5. DeepseekVL2和Gemma-3-27B模型不需要测试

### 高级配置

每个脚本都支持一些高级配置选项，如：

- `--max_tokens`: 设置模型生成的最大token数
- `--temperature`: 设置模型生成的随机性
- `--max_concurrent_requests`: 设置最大并发请求数
- `--api_key`: 手动指定API密钥（覆盖默认值）

详细配置选项请参考各脚本的帮助信息：

```bash
python <脚本名>.py --help
```

## 通过vlmevalkit框架运行
```bash
CUDA_VISIBLE_DEVICE=xxx python run.py --model xxx --data 'xx' --task xxx
```
task可以选择：cla/seg/report/measurement
model支持：
- GPT4o_MINI
- GPT4o_20240806
- Gemini2.0-pro-exp-02-5
- Doubao
- Claude
- qwen2.5-vl-72b-instruct
- QWenVL_20250125
- qwen2.5-vl-32b-instruct 
- qwen2.5-vl-7b-instruct
- qwen2.5-vl-3b-instruct
- VisualGLM_6b 
- MiniGPT-med 
- instructblip_7b
- llava_v1.5_13b 
- InternVL2_5-8B-MPO
- InternVL2_5-8B
- deepseek_vl2_small
- flamingov2
- radfm
### 配置模型
模型配置可在vlmeval/config.py的460行的supported_VLM中修改

### 配置数据集
将数据集处理成csv形式后，可在对应数据集文件中进行修改
- cla: vlmeval/dataset/image_cla.py
- seg: vlmeval/dataset/image_seg.py
- report: vlmeval/dataset/image_report.py
- measurement: vlmeval/dataset/image_measurement.py

### 测试
测试代码/vlmeval/eval_func