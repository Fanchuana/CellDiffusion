#!/bin/bash
#SBATCH --job-name=squidiff_hepg2_total_film  # 作业名称
#SBATCH --output=/work/home/cryoem666/xyf/temp/pycharm/Squidiff/script/log/squidiff_hepg2_total_film.out  # 输出文件
#SBATCH --error=/work/home/cryoem666/xyf/temp/pycharm/Squidiff/script/log/squidiff_hepg2_total_film.err   # 错误输出文件
#SBATCH --ntasks=1                        # 总任务数量
#SBATCH --cpus-per-task=12                 # 每个任务的CPU核数
#SBATCH --gres=gpu:1                      # 申请 4 个 GPU
#SBATCH --mem=50G                         # 分配的内存
#SBATCH --time=10-00:00:00                # 最长运行时间，格式为 HH:MM:SS
#SBATCH --partition=normal               # 使用的分区（指定为 normal）
#SBATCH --nodelist=g01n14               # 指定节点为 g01n01
#!/bin/bash
# train_and_eval_robust.sh

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

# 配置参数
steps=0
network_name="cfg_prob_0.15"
actual_steps=$((steps * 1000))
output_dir="ablation_study/network_result"
checkpoint_dir="/work/home/cryoem666/xyf/temp/pycharm/Squidiff/Squidiff/checkpoints/network_ablation/squidiff_hepg2_total_cfg_prob_0.15_steps_1500k"
log_dir="logs"
config_file="/work/home/cryoem666/xyf/temp/pycharm/state/gene_perturnb_state/config_toml/hepg2.toml"
inference_script="/work/home/cryoem666/xyf/temp/pycharm/Squidiff/script/script_group_replogle/normal_inference_scripts/inference_hepg2.sh"
experiment_name="squidiff_hepg2_total_${network_name}_steps_${steps}k"
# 打印配置
print_config() {
    echo "========================================"
    echo "         Squidiff 训练和评估脚本"
    echo "========================================"
    echo " 实验名称: ${experiment_name} "
    echo " 使用的网络类型: ${network_name} "
    echo "配置参数:"
    echo "  数据集: Hepg2"
    echo "  Steps: ${steps}k"
    echo "  实际训练步数: ${actual_steps}"
    echo "  输出目录: ${output_dir}"
    echo "  检查点目录: ${checkpoint_dir}"
    echo "  日志目录: ${log_dir}"
    echo "  配置文件: ${config_file}"
    echo "  推理脚本: ${inference_script}"
    echo "========================================"
}

# 检查必要的目录和文件
check_prerequisites() {
    echo "检查依赖..."
    
    # 检查配置文件
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件不存在: $config_file"
        return 1
    fi
    
    # 检查推理脚本
    if [ ! -f "$inference_script" ]; then
        echo "错误: 推理脚本不存在: $inference_script"
        return 1
    fi
    
    # 创建必要的目录
    mkdir -p "${log_dir}/${experiment_name}"
    mkdir -p "${checkpoint_dir}"
    mkdir -p "${output_dir}"
    
    echo "依赖检查通过"
    return 0
}

# 训练函数
run_training() {
    local start_time=$(date +"%Y年%m月%d日_%H时%M分%S秒")
    echo "训练开始: $start_time"
    
    # 加载CUDA模块
    if command -v module &> /dev/null; then
        module load cuda11.8
        echo "已加载CUDA 11.8模块"
    else
        echo "警告: module命令不可用，跳过CUDA加载"
    fi
    
    # 激活环境
    source activate my_state
    #随机分配一个端口号
    export MASTER_PORT=$((RANDOM%10000+20000))
    echo "使用的MASTER_PORT: $MASTER_PORT"
    
    # 切换到工作目录
    cd /work/home/cryoem666/xyf/temp/pycharm/Squidiff/Squidiff || {
        echo "错误: 无法切换到工作目录"
        return 1
    }
    
    echo "当前目录: $(pwd)"
    
    # 运行训练
    python train_squidiff_ours.py \
        --logger_path "${log_dir}/${experiment_name}" \
        --resume_checkpoint "${checkpoint_dir}/${experiment_name}" \
        --toml_config "$config_file" \
        --output_dim 203 \
        --num_layers 3 \
        --batch_size 64 \
        --microbatch -1 \
        --lr 1e-4 \
        --diffusion_steps 1000 \
        --lr_anneal_steps "$actual_steps" \
        --log_interval 1000 \
        --save_interval 10000 \
        --use_vae True \
        --film True
    
    local train_status=$?
    local end_time=$(date +"%Y%m月%d日_%H时%M分%S秒")
    
    if [ $train_status -eq 0 ]; then
        echo "训练成功完成!"
        echo "开始时间: $start_time"
        echo "结束时间: $end_time"
    else
        echo "错误: 训练失败，退出码: $train_status"
        return $train_status
    fi
    
    return 0
}

# 推理和评估函数
run_inference() {
    echo "开始推理和评估..."
    
    # 创建输出目录
    local inference_output="${output_dir}/${experiment_name}"
    mkdir -p "$inference_output"
    
    echo "推理输出目录: $inference_output"
    echo "检查点路径: ${checkpoint_dir}/${experiment_name}"
    
    # 运行推理脚本
    bash "$inference_script" \
        "/work/home/cryoem666/xyf/temp/pycharm/Squidiff/Squidiff/checkpoints/network_ablation/squidiff_hepg2_total_cfg_prob_0.15_steps_1500k" \
        "/work/home/cryoem666/xyf/temp/pycharm/Squidiff/${inference_output}"
    
    local inference_status=$?
    
    if [ $inference_status -eq 0 ]; then
        echo "推理和评估成功完成!"
    else
        echo "警告: 推理脚本返回非零退出码: $inference_status"
    fi
    
    return $inference_status
}

# 主函数
main() {
    print_config
    
    # 检查依赖
    if ! check_prerequisites; then
        echo "错误: 依赖检查失败"
        exit 1
    fi
    
    # 运行训练
    #echo ""
    #echo "=== 开始训练 ==="
    #if ! run_training; then
    #    echo "错误: 训练失败，跳过推理"
    #    exit 1
    #fi
    
    # 运行推理
    echo ""
    echo "=== 开始推理 ==="
    run_inference
    
    echo ""
    echo "========================================"
    echo "        所有任务完成!"
    echo "========================================"
    echo "训练检查点: ${checkpoint_dir}/${experiment_name}"
    echo "评估结果: ${output_dir}/${experiment_name}"
    echo "========================================"
}

# 运行主函数
main "$@"

#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-1.5B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-0.5B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-3B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-7B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_EvoInstruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3.2/llama3.2_1B_instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3.2/llama3.2_3B_instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_125M/" Base Model
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_1.3B/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3/"

