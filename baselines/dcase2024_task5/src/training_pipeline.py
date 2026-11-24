import json
import os
import time
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config):

    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Stage wall-clock times, written out at the end for reporting. Setup is
    # tracked separately because it is not negligible: building the datamodule
    # also computes or loads the AFRI and per-class energy profiles, which is
    # hours when cold and under a second when cached.
    timings = {}

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    _t = time.perf_counter()
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    timings["setup_data_s"] = time.perf_counter() - _t
    log.info(f"Datamodule ready in {timings['setup_data_s']:.1f}s")

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    _t = time.perf_counter()
    model: LightningModule = hydra.utils.instantiate(config.model)
    timings["setup_model_s"] = time.perf_counter() - _t

    # Connect frequency importance tracker from model to datamodule (Phase 3)
    if hasattr(model, 'freq_importance_tracker') and model.freq_importance_tracker is not None:
        if hasattr(datamodule, 'set_freq_importance_tracker'):
            datamodule.set_freq_importance_tracker(model.freq_importance_tracker)
            log.info("✓ Frequency importance tracker connected from model to datamodule")

    load_weight_from = config.train_param.get("load_weight_from")
    if load_weight_from:
        if not os.path.isabs(load_weight_from):
            load_weight_from = os.path.join(
                hydra.utils.get_original_cwd(), load_weight_from
            )
        model = model.load_from_checkpoint(load_weight_from)
        print("Reload model weights from %s." % load_weight_from)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        _t = time.perf_counter()
        trainer.fit(model=model, datamodule=datamodule)
        timings["train_s"] = time.perf_counter() - _t
        timings["train_epochs"] = int(trainer.current_epoch) + 1
        log.info(f"Training finished in {timings['train_s'] / 60:.1f} min "
                 f"over {timings['train_epochs']} epoch(s)")

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    def cleanup_after_training(model):
        import gc
        import torch
        model.cpu() 
        torch.cuda.empty_cache()
        gc.collect()
        log.info("GPU memory cleared after training")

    # Test the model
    if config.get("test"):
        # if(config.trainer.limit_train_batches is not None and config.trainer.limit_train_batches <= 500):
        #     ckpt_path = config.trainer.resume_from_checkpoint
        # else:
        cleanup_after_training(model)
        ckpt_path = "best"
        # ckpt_path = "/vol/research/dcase2022/project/hhlab/logs/experiments/runs/prototype_network/2022-05-18_21-38-46-with_validation_5/checkpoints/epoch_038_val_acc_0.95.ckpt"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        _t = time.perf_counter()
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
        )
        timings["test_s"] = time.perf_counter() - _t
        # Per-file inference cost is the comparable number across configurations:
        # one test "batch" is one whole recording.
        try:
            n_files = len(datamodule.data_test)
            if n_files:
                timings["test_files"] = int(n_files)
                timings["test_s_per_file"] = timings["test_s"] / n_files
        except Exception:
            pass
        log.info(f"Testing finished in {timings['test_s'] / 60:.1f} min"
                 + (f" ({timings['test_s_per_file']:.1f}s per file)"
                    if "test_s_per_file" in timings else ""))

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Persist the stage timings next to the original working directory, keyed by
    # exp_name so run_ablation_experiments.py can collect them without parsing
    # stdout. Hydra has chdir'd into the run directory by now, hence the join.
    try:
        timings_dir = os.path.join(config.original_work_dir, "logs", "timings")
        os.makedirs(timings_dir, exist_ok=True)
        timings["exp_name"] = config.get("exp_name")
        timings["total_s"] = sum(
            v for k, v in timings.items() if k.endswith("_s") and k != "total_s"
        )
        with open(os.path.join(timings_dir, f"{config.get('exp_name')}.json"),
                  "w") as handle:
            json.dump(timings, handle, indent=2)
        log.info("Stage timings: " + ", ".join(
            f"{k}={v:.1f}s" for k, v in timings.items() if k.endswith("_s")))
    except Exception as exc:
        log.warning(f"Could not write stage timings: {exc}")

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score


