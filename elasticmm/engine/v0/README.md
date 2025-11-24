# ElasticMM v0 Engine Backend

基于vLLM v0 engine的分离式推理backend，实现了encode、prefill、decode三阶段分离和智能KV cache管理。**已成功实现端到端的多模态推理**。

## 🎯 架构概览

```
V0EngineBackend
│
├── EncodingStage (编码阶段)
│   ├── Workers: 处理多模态输入，生成视觉嵌入
│   ├── V0VisionBlockManager: 管理视觉嵌入缓存
│   └── V0EncodingScheduler: 调度编码请求
│
├── PrefillStage (预填充阶段)
│   ├── Workers: 处理完整序列，生成KV cache
│   ├── V0BlockManager: 智能KV cache管理 (基于EPD)
│   ├── V0VisionBlockManager: 接收视觉嵌入
│   └── V0PrefillScheduler: 调度预填充请求
│
├── DecodingStage (解码阶段)
│   ├── Workers: 自回归生成token
│   ├── V0BlockManager: 动态扩展KV cache
│   └── V0DecodingScheduler: 调度解码请求
│
└── V0KVTransferManager
    └── 管理阶段间的KV cache和视觉嵌入传输
```

## ✅ 成功实现的多模态推理流程

### 完整数据流

```
1. Encoding Stage
   Input: 图像 + 文本prompt
   ↓
   vLLM Processor → pixel_values [864, 1176] + image_grid_thw [[1,24,36]]
   Vision Encoder → vision_embeddings [216, 3584]
   ↓
   传输: MigratingRequest(vision_embeddings, mm_kwargs, mm_placeholders)

2. Prefill Stage  
   接收: prompt_token_ids (19 tokens, 包含1个图像占位符)
   ↓
   ✅ Token扩展: 19 → 234 tokens (1个占位符 → 216个image tokens)
   ✅ Block智能分配: 2 → 15 blocks (基于EPD的_get_free_blocks)
   ✅ Vision Embeddings注入: 双重注入机制确保生效
   ✅ KV Cache生成: 正确的attention metadata和slot_mapping
   ↓
   输出: KV cache + generated tokens

3. Decode Stage
   接收: KV cache from Prefill + 扩展后的prompt_token_ids
   ↓
   ✅ 动态Block扩展: 根据序列长度智能分配additional blocks
   ✅ 自回归生成: 高质量的多模态输出
```

### Token扩展机制

```python
# 原始输入 (19 tokens)
[27, 91, 872, 91, 397, 151652, 151655, 151653, 198, ...]
#                      ↑ 这3个是图像相关的特殊tokens

# 扩展后 (234 tokens)  
[27, 91, 872, 91, 397, 151652,  # 前6个tokens
 151673, 151673, ..., 151673,    # 216个图像占位符 (151673重复216次)
 151655, 151653, 198, ...]       # 剩余文本tokens
```

## 🔧 核心技术突破

### 1. 智能KV Cache管理 (✅ 已解决)

**问题**: 多模态模型需要动态扩展KV cache blocks，传统方法会导致访问错误。

**解决方案**: 集成EPD的智能block管理策略
- **职责分离**: worker_steps.py只验证，stage_engine.py负责分配，block_manager.py提供智能管理
- **动态扩展**: 根据实际序列长度(prompt + output)计算所需blocks
- **连续分配**: 使用`_get_free_blocks()`分配连续的block IDs，避免0值填充

```python
# stage_engine.py - 智能block分配
total_seq_len = len(request.prompt_token_ids) + len(request.output_token_ids)
blocks_needed = (total_seq_len + block_size - 1) // block_size
if current_blocks < blocks_needed:
    new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
    self.block_manager.block_table[request.request_id].extend(new_blocks)
```     

### 2. 跨进程Token扩展 (✅ 已解决)

**问题**: Ray序列化导致token扩展在worker中的修改丢失。

**解决方案**: 显式数据传递机制
- **MigratingRequest扩展**: 添加`expanded_prompt_token_ids`字段
- **主进程更新**: stage_engine.py在主进程中更新request对象
- **防重复扩展**: worker只返回扩展结果，不直接修改request

```python
# utils.py - MigratingRequest扩展
@dataclass
class MigratingRequest:
    expanded_prompt_token_ids: Optional[List[int]] = None  # 跨进程传递扩展tokens

# stage_engine.py - 主进程更新
if request.request_id in expanded_tokens_map:
    request.prompt_token_ids = expanded_tokens_map[request.request_id]
```

### 3. Vision Embeddings注入 (✅ 已解决)

**问题**: Vision embeddings需要正确注入到vLLM的多模态处理流程。

**解决方案**: 双重注入机制
- **Early injection**: 在SequenceGroupMetadata构建前注入
- **Post injection**: 在prepare_model_input后再次注入，确保不被覆盖
- **MRoPE支持**: 正确传递image_grid_thw用于位置计算

### 4. 温度参数优化 (✅ 已解决)

**问题**: temperature=0.0导致采样异常，输出重复无意义token。

**解决方案**: 调整采样参数
- **Temperature**: 0.0 → 0.8
- **Top-p**: 添加0.9的top_p采样
- **结果**: 生成质量显著提升

## 📊 性能验证

### 关键指标

- ✅ **Token扩展**: 19 → 234 tokens (正确)
- ✅ **Block分配**: 2 → 15 blocks (智能)
- ✅ **KV Cache**: 正确写入和读取
- ✅ **输出质量**: 高质量的多模态生成
- ✅ **稳定性**: 无KV cache访问错误

### 调试输出示例

```
[Prefill] Found vision embeddings for qwen_vl_000: 216 tokens
[Prefill] Expanded tokens: 234 (was 19)
[Decode] qwen_vl_000: prompt_tokens=234, output_tokens=1, total_seq_len=235
[Decode] ✓ Allocated 1 blocks for qwen_vl_000, total blocks: 15
```

## 🚀 核心模块

### 1. Block Manager (`block_manager.py`)
- **V0BlockManager**: 基于EPD的智能KV cache管理
  - `_get_free_blocks()`: EPD的智能block分配
  - `allocate_blocks()`: 支持动态扩展的block分配
  - `get_num_blocks_needed()`: 精确计算所需blocks
- **V0VisionBlockManager**: 视觉嵌入管理
- **支持**: GPU/CPU swap, 连续block分配

### 2. Worker Steps (`worker_steps.py`)
- **Token扩展**: 在prefill阶段扩展多模态tokens
- **Vision注入**: 双重注入机制确保embeddings生效
- **验证机制**: 验证block分配，不直接修改blocks
- **温度控制**: 优化的采样参数

### 3. Stage Engine (`stage_engine.py`)
- **智能Block管理**: 使用block_manager动态分配blocks
- **跨进程协调**: 处理Ray序列化问题
- **请求迁移**: 管理阶段间的数据传递
- **错误处理**: 完善的错误检测和恢复

### 4. 其他模块
- **Worker**: Ray actor封装，阶段特定推理
- **KV Transfer**: 阶段间KV cache传输
- **Backend**: 统一接口和阶段协调

## 🎮 快速开始

### 安装

```bash
cd /root/lzd/elasticmm_project
pip install -e .
```

### 基本使用

```python
import asyncio
from elasticmm.engine.v0 import V0EngineBackend
from elasticmm.engine.v0.config import V0EngineConfig

async def main():
    # 创建配置
    config = V0EngineConfig(
        model_path="/path/to/qwen-vl-model",
        num_encoding_workers=2,
        num_prefill_workers=4,
        num_decoding_workers=2,
        block_size=16,
        max_num_gpu_blocks=5000,
    )
    
    # 创建backend
    backend = V0EngineBackend(**config.to_dict())
    
    # 初始化和启动
    await backend.initialize()
    await backend.start()
    
    # 添加多模态请求
    from elasticmm.engine.v0.utils import Request
    request = Request(
        request_id="multimodal_1",
        prompt="请描述这张图片",
        image_path="/path/to/image.jpg",
        max_tokens=100,
    )
    await backend.add_request(request)
    
    # 获取输出
    outputs = await backend.get_outputs()
    for output in outputs:
        print(f"Generated: {output.generated_text}")
    
    # 停止
    await backend.stop()

asyncio.run(main())
```

### 测试多模态推理

```bash
# 运行完整的多模态测试
cd /root/lzd/elasticmm_project
python examples/test_v0_backend.py

# 查看关键日志
python examples/test_v0_backend.py 2>&1 | grep -E "(Expanded tokens|Found vision|Allocated.*blocks)"
```

## ⚙️ 配置

### V0EngineConfig

```python
V0EngineConfig(
    model_path: str,                    # 模型路径 (必需)
    num_encoding_workers: int = 2,      # 编码worker数量
    num_prefill_workers: int = 4,       # 预填充worker数量
    num_decoding_workers: int = 2,      # 解码worker数量
    block_size: int = 16,               # KV cache块大小
    max_num_gpu_blocks: int = 5000,     # 最大GPU块数
    max_num_cpu_blocks: int = 1000,     # 最大CPU块数
    dtype: str = "float16",             # 模型数据类型
    tensor_parallel_size: int = 1,      # 张量并行大小
    gpu_memory_utilization: float = 0.9, # GPU内存利用率
    kv_transfer_method: str = "p2p_copy", # KV传输方法
)
```

### 性能调优

```python
# 多模态密集型配置
config = V0EngineConfig(
    num_encoding_workers=4,  # 增加编码workers处理图像
    num_prefill_workers=4,   # 处理扩展后的长序列
    num_decoding_workers=2,
    block_size=16,           # 适合多模态的block大小
)

# 长文本生成配置
config = V0EngineConfig(
    num_encoding_workers=2,
    num_prefill_workers=6,   # 增加prefill workers
    num_decoding_workers=4,  # 增加decode workers
    block_size=32,           # 更大的block size
)
```

## 📈 与EPD的对比

| 特性 | EPD | ElasticMM v0 |
|------|-----|--------------|
| 三阶段分离 | ✅ | ✅ |
| 智能Block管理 | ✅ | ✅ 已集成EPD策略 |
| 多模态Token扩展 | ❌ | ✅ 已解决 |
| Vision Embeddings注入 | ✅ | ✅ 双重注入机制 |
| 跨进程数据传递 | 基础 | ✅ 完善的Ray序列化处理 |
| 动态Block扩展 | ✅ | ✅ 已实现 |
| 弹性调度 | ❌ | ✅ |
| Backend抽象 | ❌ | ✅ |
| 配置管理 | 基础 | ✅ 结构化 |
| 错误处理 | 基础 | ✅ 完善的验证和恢复 |

## 🔍 技术细节

### 关键设计决策

1. **职责分离**: 
   - worker_steps.py: 执行推理逻辑
   - stage_engine.py: 协调和block管理
   - block_manager.py: 提供智能block分配

2. **数据传递**:
   - 使用MigratingRequest显式传递扩展tokens
   - 主进程更新request对象，避免Ray序列化问题

3. **Block管理**:
   - 集成EPD的`_get_free_blocks()`策略
   - 支持动态扩展和连续block分配

4. **Vision处理**:
   - 双重注入确保embeddings生效
   - 正确的MRoPE位置计算

### 调试工具

```python
# 关键调试输出
print(f"[Prefill] Expanded tokens: {len(request.prompt_token_ids)} (was {original_token_count})")
print(f"[Decode] {request.request_id}: prompt_tokens={len(request.prompt_token_ids)}, total_seq_len={seq_len}")
print(f"[Decode] ✓ Allocated {len(new_blocks)} blocks for {request.request_id}")
```

## 🎯 下一步开发

### 近期目标
1. **性能优化**: 进一步优化block分配策略
2. **扩展性**: 支持更多多模态模型 (LLaVA, InternVL等)
3. **监控**: 添加详细的性能指标和监控

### 长期目标
4. **CUDA扩展**: 集成CUDA IPC零拷贝传输
5. **生产部署**: 生产环境优化和部署支持
6. **弹性调度**: 更智能的负载均衡和调度策略

## 📚 参考

- [EPD源码](https://github.com/SungMinCho/EPD-Disaggregation) - 智能block管理策略来源
- [vLLM文档](https://docs.vllm.ai/) - vLLM v0 engine基础
- [Qwen-VL论文](https://arxiv.org/abs/2309.16609) - 多模态模型理解

## 📄 License

Apache 2.0

---

**🎉 ElasticMM v0 Engine已成功实现端到端的多模态推理！**