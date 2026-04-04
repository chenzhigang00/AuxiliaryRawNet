import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.RawSNet import RawSNet
from datasets.TrainSpeechDataset import get_dataset
from datasets.EvalSpeechDataset import get_dataset as get_eval_dataset

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False


def parse_cli_args(argv):
    mode = "train"
    filtered_argv = []
    i = 0

    while i < len(argv):
        arg = argv[i]
        if arg == "--mode":
            if i + 1 >= len(argv):
                raise ValueError("--mode requires a value: train or eval")
            mode = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
            continue

        filtered_argv.append(arg)
        i += 1

    aliases = {"train": "train", "eval": "eval", "evaluate": "eval"}
    if mode not in aliases:
        raise ValueError("--mode must be one of: train, eval")

    return aliases[mode], filtered_argv


def prepare_run_opts(run_opts):
    requested_device = run_opts.get("device", "cuda:0")

    if requested_device.startswith("npu"):
        if not HAS_TORCH_NPU:
            raise RuntimeError(
                "NPU device requested but torch_npu is not installed. "
                "Install torch-npu and source /usr/local/Ascend/ascend-toolkit/set_env.sh first."
            )
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError(
                "NPU device requested but torch.npu is not available. "
                "Check the Ascend runtime and source /usr/local/Ascend/ascend-toolkit/set_env.sh."
            )
        return run_opts

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available():
            run_opts["device"] = "npu:0"
        else:
            run_opts["device"] = "cpu"

    return run_opts


def empty_accelerator_cache():
    if HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_train(hparams_file, run_opts, overrides):
    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory.
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = get_dataset(hparams)
    spk_id_brain = RawSNet(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    spk_id_brain.fit(
        epoch_counter=spk_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )


def run_eval(hparams_file, run_opts, overrides):
    print("Start to Inference")
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    saved_hparams_file = os.path.join(hparams["output_folder"], "hyperparams.yaml")
    if not os.path.exists(saved_hparams_file):
        raise FileNotFoundError(
            "Trained experiment config not found at {}. Run training first.".format(
                saved_hparams_file
            )
        )

    with open(saved_hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["batch_size"] = 64
    hparams["dataloader_options"]["batch_size"] = 16
    datasets = get_eval_dataset(hparams)
    os.makedirs("predictions", exist_ok=True)

    spk_id_brain = RawSNet(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    spk_id_brain.evaluate(
        test_set=datasets["eval"],
        min_key="eer",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )


if __name__ == "__main__":
    empty_accelerator_cache()
    mode, filtered_argv = parse_cli_args(sys.argv[1:])

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(filtered_argv)
    run_opts = prepare_run_opts(run_opts)
    print("Mode:", mode)
    print("Using device:", run_opts["device"])

    if mode == "train":
        run_train(hparams_file, run_opts, overrides)
    else:
        run_eval(hparams_file, run_opts, overrides)
