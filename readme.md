
# U2-Bench  
A Multimodal Evaluation Framework Built on VLMEvalKit  


## Deployment Instructions for Selected Models  

| Script File                  | Applicable Models                     | Description                                                                 |  
|-----------------------------|---------------------------------------|-----------------------------------------------------------------------------|  
| `batch_test.py`             | QwenVL7B, QwenVL32B, QwenVL72B        | For testing locally deployed Tongyi Qianwen vision series models           |  
| `batch_test_xiaohu.py`      | GPT series, Claude series, Qwen max   | For testing OpenAI and Anthropic models (via Xiaohu API proxy), and Qwen max |  
| `batch_test_volc.py`        | doubao series                         | For testing Doubao models via Volcano Engine Ark Platform                  |  
| `batch_test_xiaohu_gemini25.py` | Gemini-2.5 series                  | Specifically optimized for testing Google Gemini 2.5 models                 |  
| `batch_test_gemini_fixed.py`| Gemini-2.0 series                     | Specifically optimized for testing Google Gemini 2.0 models                 |  
| `batch_test_dashscope.py`   | qwen2.5-vl-3b                         | For testing qwen2.5-vl-3b via Alibaba Cloud Bai Lian Platform             |  

**All script files are located in `vlmeval/silicon_test`.**  


### Common Features  
All scripts support the following basic functionalities:  
1. **Batch Testing**: Perform batch testing across multiple datasets and tasks.  
2. **Multimodal Input**: Support combined input of images and text.  
3. **Result Saving**: Automatically save test results to specified directories.  
4. **Asynchronous Processing**: Support asynchronous API calls to improve testing efficiency.  


### Usage Guide  

#### 1. Basic Usage  
Each script supports the following command-line arguments:  
```bash  
python <script_name>.py --task_type <task_type> --dataset <dataset_name> --output_dir <output_directory>  
```  
- `<task_type>`: Optional values: `cla` (classification), `measurement` (measurement), `seg` (segmentation), `report` (report generation).  
- `<dataset_name>`: Dataset number corresponding to the task type (e.g., `03`, `10`).  
- `<output_dir>`: Result saving directory (default: `outputs/dolphin-output`).  

#### 2. Model-Specific Testing Examples  
##### Test QwenVL Series Models  
```bash  
# Test QwenVL7B on classification task with dataset 03  
python batch_test.py --task_type cla --dataset 03 --model Qwen2.5-VL-7B-Instruct-Pro  
```  

##### Test GPT-4o Models  
```bash  
# Test GPT-4o on segmentation task with dataset 04  
python batch_test_xiaohu.py --task_type seg --dataset 04 --model gpt-4o-2024-08-06  
```  

##### Test Doubao Models  
```bash  
# Test Doubao on report generation task with dataset 10  
python batch_test_volc.py --task_type report --dataset 10 --model doubao-1.5-vision-pro-32k-250115  
```  

##### Test Gemini 2.5 Models  
```bash  
# Test Gemini 2.5 on measurement task with dataset 27  
python batch_test_xiaohu_gemini25.py --task_type measurement --dataset 27  
```  

#### 3. Test All Datasets  
Use the `--test_all_datasets` flag to test all datasets under a specified task type:  
```bash  
python batch_test.py --task_type cla --test_all_datasets --model Qwen2.5-VL-7B-Instruct-Pro  
```  

#### 4. Parallel Multi-Task Testing  
Some scripts support parallel testing of multiple tasks and datasets:  
```bash  
python batch_test.py --multi_task --task_types cla seg --datasets 03 04  
```  


### Output Structure  
Test results are saved in the specified output directory with the following structure:  
```  
outputs/dolphin-output/  
├── <task_type>/  
│   ├── <dataset>/  
│   │   ├── <model_name>/  
│   │   │   ├── multimodal_test_results_<timestamp>.jsonl  # Raw test results  
│   │   │   └── <model_name>_combined_results.jsonl        # Combined results  
```  


### Notes  
1. Ensure correct setup of API keys or model access credentials.  
2. For local model testing, ensure models are correctly loaded and accessible.  
3. Network issues may cause API failures; scripts will automatically retry.  
4. For large datasets, use `--sample_limit` to restrict the number of test samples.  
5. DeepseekVL2 and Gemma-3-27B models do not require testing.  


### Advanced Configuration  
Each script supports advanced configuration options, such as:  
- `--max_tokens`: Set maximum tokens for model generation.  
- `--temperature`: Control model generation randomness.  
- `--max_concurrent_requests`: Set maximum concurrent API requests.  
- `--api_key`: Manually specify an API key (overrides defaults).  

For detailed options, check the script help:  
```bash  
python <script_name>.py --help  
```  


## Running via the VLMEvalKit Framework  
```bash  
CUDA_VISIBLE_DEVICE=xxx python run.py --model xxx --data 'xx' --task xxx  
```  
- **Task Options**: `cla`/`seg`/`report`/`measurement`  
- **Supported Models**:  
  - GPT4o_MINI, GPT4o_20240806  
  - Gemini2.0-pro-exp-02-5  
  - Doubao, Claude  
  - qwen2.5-vl-72b-instruct, QWenVL_20250125  
  - qwen2.5-vl-32b-instruct, qwen2.5-vl-7b-instruct, qwen2.5-vl-3b-instruct  
  - VisualGLM_6b, MiniGPT-med, instructblip_7b  
  - llava_v1.5_13b, InternVL2_5-8B-MPO, InternVL2_5-8B  
  - deepseek_vl2_small, flamingov2, radfm  


### Model Configuration  
Modify model configurations in `vlmeval/config.py` at line 460 in `supported_VLM`.  


### Dataset Configuration  
Process datasets into CSV format and modify corresponding dataset files:  
- Classification: `vlmeval/dataset/image_cla.py`  
- Segmentation: `vlmeval/dataset/image_seg.py`  
- Report Generation: `vlmeval/dataset/image_report.py`  
- Measurement: `vlmeval/dataset/image_measurement.py`  


### Testing Code  
Test code is located in `vlmeval/eval_func`.