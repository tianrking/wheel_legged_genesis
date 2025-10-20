import argparse
import os
import pickle
import numpy as np
import torch
import copy
import sys

# ------------------- 关键修改部分开始 -------------------

# 1. **修正路径**：这是解决当前问题的核心。
#    我们必须先将项目的根目录（'wheel_legged_genesis'）添加到 Python 的搜索路径中。
#    这样，Python 才能正确地找到 'utils' 和 'locomotion' 这两个文件夹。
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 2. **初始化引擎**：现在路径正确了，我们按原计划初始化 Genesis。
import genesis as gs
gs.init(backend=gs.gpu)

# 3. **导入项目模块**：在路径修正和引擎初始化之后，现在这些导入语句可以正常工作了。
from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner
from utils import gamepad

# -------------------- 关键修改部分结束 --------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("--ckpt", type=int, default=7300)
    args = parser.parse_args()
    
    log_dir = f"logs/{args.exp_name}"
    
    # 加载训练时保存的配置文件
    try:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    except FileNotFoundError:
        print(f"错误：找不到配置文件 'logs/{args.exp_name}/cfgs.pkl'。")
        print("请确保你已经成功训练了一个模型，并且指定的实验名称是正确的。")
        return

    # --- 你可以在这里修改配置进行实验 ---
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym" # 可选: "agent_train_gym" 或 "circular"
    # ------------------------------------

    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        robot_morphs="urdf",
        show_viewer=True,
        num_view = 1,
        train_mode=False
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    
    try:
        runner.load(resume_path)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{resume_path}'。")
        print(f"请检查指定的检查点 (--ckpt {args.ckpt}) 是否存在。")
        return

    # JIT编译模型以加速推理
    print("\n--- 正在编译模型以加速... ---")
    model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
    jit_model_path = os.path.join(log_dir, "policy_jit.pt")
    torch.jit.script(model).save(jit_model_path)
    
    print("\n--- 正在加载加速后的模型 ---")
    try:
        loaded_policy = torch.jit.load(jit_model_path)
        loaded_policy.eval() # 设置为评估模式
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    obs, _ = env.reset()
    pad = gamepad.control_gamepad(command_cfg,[env.command_cfg["lin_vel_x_range"][1],
                                               env.command_cfg["lin_vel_y_range"][1],
                                               env.command_cfg["ang_vel_range"][1],
                                               0.05, 0.05, 1.0])
    
    print("\n--- 开始仿真 ---")
    print("你可以使用键盘或手柄来控制机器人。")
    
    with torch.no_grad():
        while True:
            actions = loaded_policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            comands, reset_flag = pad.get_commands()
            env.set_commands(np.arange(env.num_envs), comands)
            if reset_flag:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()

