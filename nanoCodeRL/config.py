"""nanoCodeRL configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen3.5-4B"
    load_in_4bit: bool = True  # QLoRA; set False for full BF16 on A100

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )

    # Generation
    enable_thinking: bool = False
    max_completion_length: int = 2048  # tokens (RTX 5090 32GB)
    temperature: float = 0.7
    top_p: float = 0.95

    # DAPO / GRPO
    num_rollouts: int = 8        # rollouts per prompt
    batch_size: int = 8          # prompts per step (RTX 5090 32GB; use 16 on A100 80GB)
    num_train_steps: int = 200
    learning_rate: float = 1e-6
    lr_scheduler: str = "cosine"
    warmup_steps: int = 10
    kl_coef: float = 0.0         # DAPO uses Clip-Higher instead of KL penalty
    clip_eps: float = 0.2        # standard PPO clip
    clip_eps_high: float = 0.28  # DAPO Clip-Higher (asymmetric upward)

    # Reward / Sandbox
    sandbox_timeout: int = 5     # seconds per code execution
    max_test_cases: int = 50     # max unit tests to run per problem
    sandbox_max_workers: int = 8  # parallel sandbox processes
    use_docker_sandbox: bool = False  # use Docker for sandboxed execution

    # Data — train on CodeContests, eval on HumanEval + MBPP (no overlap)
    train_benchmarks: list[str] = field(
        default_factory=lambda: ["code_contests"]
    )
    # CodeContests difficulty filter. Enum clusters: 0=unknown(33%), 1-6=easy(7%),
    # 7-11=medium/hard(52%), 12+=very hard(8%). Default 7 keeps easy problems only.
    # Set None to disable.
    cc_max_difficulty: int | None = 7
    # Primary filter: skip problems where shortest accepted solution > N chars.
    # 800 chars ≈ 250 tokens — ensures completions fit within max_completion_length.
    # Set None to disable.
    cc_max_solution_chars: int | None = 800

    # Evaluation
    eval_every_n_steps: int = 50
    eval_benchmarks: list[str] = field(
        default_factory=lambda: ["humaneval", "mbpp"]
    )
    eval_subset_size: int = 30  # problems per benchmark for intermediate eval (0 = full)

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "nanoCodeRL"
