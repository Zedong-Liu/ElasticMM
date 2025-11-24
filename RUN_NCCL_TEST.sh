#!/bin/bash
# 测试NCCL预留ranks修复

echo "=================================="
echo "NCCL预留Ranks修复测试"
echo "=================================="

cd /root/lzd/elasticmm_project

# 清理之前的Ray实例
ray stop --force 2>/dev/null || true
sleep 2

# 运行测试
timeout 120 python examples/test_nccl_fix.py 2>&1 | tee test_nccl_output.log

echo ""
echo "=================================="
echo "测试完成，输出保存到 test_nccl_output.log"
echo "=================================="





