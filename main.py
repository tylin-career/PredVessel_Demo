import os
import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta
from functools import partial
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import yaml

from cseta.common.logger import logger, setup_logger, setup_ddp_logger
from cseta.common.utils import save_pickle
from cseta.models.nn.deeprnn import DeepRNN
from cseta.models.nn.helper import set_seed
from cseta.models.nn.hyper_param_tuning import hyper_parameter_tuning
from cseta.models.nn.read_model import read_historical_model_outputs
from cseta.models.nn.model_config import (
    ModelConfiguration,
    ModelConfigurationPSA,
    ModelConfigurationKeeLung,
    WaitingTimeModelConfiguration,
)
from cseta.models.nn.trainer import Trainer
from cseta.preprocess.nn.nn_preprocess import preprocess, postprocess
from cseta.preprocess.nn.sequence_generator import SequenceGenerator, pad_collate

# from azureml.core.run import Run
import mlflow

from cseta.common.util.s3_conn import S3Connection
from cseta.common.util.eval_logging import (
    get_summary_by_prediction_moments,
    plot_summary,
    writer_log,
)

current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
GVVMC_HOME = os.environ.get("GVVMC_HOME")
DATA_DIR = Path(GVVMC_HOME) / "data" if GVVMC_HOME else None
MODEL_DIR = DATA_DIR / "models" if DATA_DIR else None
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(GVVMC_HOME) / "log" / "nn" if GVVMC_HOME else None
S3_CONFIG_DIR = (
    Path(GVVMC_HOME) / "credentials" / "s3_config.yml" if GVVMC_HOME else None
)

GVVMC_BUCKET = "forecast.gvvmc"


def setup(world_size=-1, local_rank=-1, global_rank=-1):
    if str(args.is_psa).lower() in ["true", "1"]:
        config = ModelConfigurationPSA()
    elif str(args.is_keelung).lower() in ["true", "1"]:
        config = ModelConfigurationKeeLung()
    elif str(args.is_waiting_time_model).lower() in ["true", "1"]:
        config = WaitingTimeModelConfiguration()
    else:
        config = ModelConfiguration()
    set_seed(config.seed)
    if args.use_azure:
        config.MODE = args.mode
        config.use_azure = True
        config.use_allegro = False
        f = (
            "processed_debug_set.pkl"
            if config.MODE == "DEBUG"
            else "processed_fullset.pkl"
        )
        config.train_data_path = os.path.join(args.train_data_path, f)
        config.test_data_path = os.path.join(args.test_data_path, f)
        config.valid_num_months = int(args.valid_num_months)
        config.output_path = "./outputs"
        config.log_path = "./logs"
    else:
        assert GVVMC_HOME is not None
        config.use_azure = False
        config.train_data_path = (
            DATA_DIR / "a5_raw_data" / config.trainset / config.file_name
        )
        config.test_data_path = (
            DATA_DIR / "a5_raw_data" / config.testset / config.file_name
        )
        config.output_path = MODEL_DIR
        config.log_path = LOG_DIR
        config.port_pair_path = (
            DATA_DIR
            / "a5_raw_data"
            / config.trainset
            / "train_port_pair_statistics.csv"
        )
        config.output_mode = args.output_mode
        config.data_src = args.data_src
        config.world_size = world_size
        config.local_rank = local_rank
        config.global_rank = global_rank

    if config.MODE == "DEBUG":
        config.epoch = 2
        config.step_freq = 4
    elif config.MODE == "SUB_SET":
        config.step_freq = int(config.step_freq / 5)

    Path(config.log_path).mkdir(parents=True, exist_ok=True)

    if local_rank != -1:
        setup_ddp_logger(
            rank=global_rank,
            log_filename=os.path.join(
                config.log_path, config.MODE.lower() + "_ddp_training.log"
            ),
        )
    else:
        setup_logger(
            log_filename=os.path.join(
                config.log_path, config.MODE.lower() + "_training.log"
            )
        )

    writer = None
    # if config.use_allegro and (global_rank == -1 or global_rank == 0):
    if config.use_allegro:
        from ds_experiment_writer.experiment import Experiment

        tags = [config.MODE]
        tags = tags + ["PSA"] if config.PSA else tags
        tags = tags + ["Keelung"] if config.keelung else tags
        tags = tags + [f"Rank: {global_rank}"] if global_rank != -1 else tags
        writer = Experiment(
            log_dir=str(config.log_path),
            data_config=config.to_dict(),
            model_config={},
            project="transit_time",
            enforce_data_range=False,
            remarks=config.remarks,
            tags=tags,
            entity="vps",
            backend="allegro",
        )
        set_seed(config.seed)
        logger.info("Experiment ID: " + writer.exp_id)
    return config, writer


def get_training_data(config, s3_conn=None):
    logger.info("Start training data preprocess.")

    if config.data_src == "s3" and s3_conn is not None:
        for path in [config.train_data_path, config.port_pair_path]:
            download_file_from_storage_grid(s3_conn, path)

    df_train = (
        pd.read_parquet(config.train_data_path)
        if str(config.train_data_path).endswith(".parquet")
        else pd.read_pickle(config.train_data_path)
    )
    logger.info(
        f"Loaded training data from {str(config.train_data_path)}, size {df_train.shape[0]}"
    )

    df_train, le, scaler = preprocess(config, df_train, "train", False, False)

    val_start = df_train["dest_ata_utc"].max() - timedelta(
        days=int(config.valid_num_months * 30)
    )
    df_val = df_train[df_train["dest_ata_utc"] > val_start].reset_index(drop=True)
    df_train = df_train[df_train["dest_ata_utc"] < val_start].reset_index(drop=True)
    logger.info(
        "Train dest_ata_utc from %s to %s, size %d"
        % (
            df_train["dest_ata_utc"].min(),
            df_train["dest_ata_utc"].max(),
            df_train.shape[0],
        )
    )
    logger.info(
        "Valid dest_ata_utc from %s to %s, size %d"
        % (df_val["dest_ata_utc"].min(), df_val["dest_ata_utc"].max(), df_val.shape[0])
    )

    train_seq = SequenceGenerator(config, df=df_train, dataset="train")
    val_seq = SequenceGenerator(config, df=df_val, dataset="train")
    return train_seq, val_seq, le, scaler


def get_testing_data(config, le, scaler, s3_conn=None):
    logger.info("Start testing data preprocess.")

    if config.data_src == "s3" and s3_conn is not None:
        download_file_from_storage_grid(s3_conn, config.test_data_path)

    df_test = (
        pd.read_parquet(config.test_data_path)
        if str(config.test_data_path).endswith(".parquet")
        else pd.read_pickle(config.test_data_path)
    )
    logger.info(
        f"Loaded testing data from {str(config.test_data_path)}, size {df_test.shape[0]}"
    )

    df_test = preprocess(config, df_test, "test", False, False, le, scaler)
    logger.info(
        "Test dest_ata_utc from  %s to %s, size %d"
        % (
            df_test["dest_ata_utc"].min(),
            df_test["dest_ata_utc"].max(),
            df_test.shape[0],
        )
    )

    test_seq = SequenceGenerator(config, df=df_test, dataset="test")
    time_spent = (datetime.now() - config.times["init_time"]).total_seconds()
    config.times["preprocess_hrs"] = "%d hr %d min" % (
        time_spent // 3600,
        time_spent % 3600 // 60,
    )
    logger.info(
        "Finished Preprocessing. Total time spent: %s" % config.times["preprocess_hrs"]
    )
    return test_seq


def download_file_from_storage_grid(s3_conn, path):
    path_str = str(path)
    s3_object_path = path_str
    s3_object_path = s3_object_path.replace(GVVMC_HOME + os.path.sep, "")
    if sys.platform == "win32":
        s3_object_path = s3_object_path.replace(os.path.sep, "/")
    s3_conn.download_file(path_str, GVVMC_BUCKET, s3_object_path)


def train(config, train_seq, val_seq, test_seq, ddp=False, writer=None):
    if torch.cuda.is_available():
        logger.info("CUDA is available")
    else:
        logger.info("CUDA is not available")

    logger.info(f"Start Training")
    start_time = time.time()
    collate_fn = partial(pad_collate, config=config, dataset="train")

    set_seed(config.seed)
    model = DeepRNN(config)

    if ddp:
        set_seed(config.seed)
        dist_sampler = DistributedSampler(dataset=train_seq)
        train_loader = DataLoader(
            train_seq,
            config.batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            sampler=dist_sampler,
        )
        val_loader = DataLoader(
            val_seq,
            config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

        set_seed(config.seed)
        trainer = Trainer(
            config,
            model,
            ddp_worker=True,
            local_rank=config.local_rank,
            global_rank=config.global_rank,
        )
        final_model = trainer.fit(train_loader, val_loader, writer, config.step_freq)
    else:
        set_seed(config.seed)
        train_loader = DataLoader(
            train_seq,
            config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_seq,
            config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_seq,
            config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

        set_seed(config.seed)
        trainer = Trainer(config, model, ddp_worker=False)
        final_model = trainer.fit(
            train_loader, val_loader, test_loader, writer, config.step_freq
        )

    time_spent = time.time() - start_time
    config.times["train_hrs"] = "%d hr %d min" % (
        time_spent // 3600,
        time_spent % 3600 // 60,
    )
    logger.info("Finished Training. Total time spent: %s" % config.times["train_hrs"])
    return final_model


def predict(config, test_seq, model):
    collate_fn = partial(pad_collate, config=config, dataset="test")
    test_loader = DataLoader(
        test_seq,
        config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    pred = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch[0]
            x = batch[1]
            output = model(x_seq, x, config.device)
            pred.append(output.cpu().data.numpy())
    return np.concatenate(pred, axis=0)


def count_invalid_predictions(config, df, summary):
    for i, (pred, is_evaluate) in enumerate(
        zip(config.meta["pred_cols"], config.meta["is_evaluate_cols"])
    ):
        negative_count = df[(df[pred] < 0) & (df[is_evaluate])].shape[0]
        summary.update({pred.replace("y_pred", "-ve"): negative_count})
        logger.info("Negative %s count: %d" % (pred, negative_count))
        if i < len(config.meta["pred_cols"]) - 1:
            next_pred = config.meta["pred_cols"][i + 1]
            stale_count = df[(df[pred] > df[next_pred]) & (df[is_evaluate])].shape[0]
            summary.update({pred.replace("y_pred", "stale"): stale_count})
            logger.info("Staled %s count: %d" % (pred, stale_count))


def run_log(df_summary, config):
    with mlflow.start_run():
        for subset in config.meta["subsets"]:
            fig = plot_summary(config, df_summary, subset)
            mlflow.log_figure(fig, f"{subset}.html")

    columns = [
        "abs_error_pilot",
        "abs_error_berth",
        "abs_error_pilot_prod",
        "abs_error_berth_prod",
    ]
    colors = ["blue", "red", "olive", "black"]
    vessels = ["Small Vessels (<3000TEU)", "Large Vessels (>=3000TEU)"]
    notes = df_summary.notes.unique()
    for i, note in enumerate(notes):
        for j, vessel in enumerate(vessels):
            temp = df_summary[
                (df_summary.notes == note) & (df_summary.is_large_vsl == j)
            ]
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            for column, color in zip(columns, colors):
                axs.plot(
                    "remaining_time_interval",
                    column,
                    data=temp,
                    color=color,
                    label=column,
                )
            axs.legend()
            axs.set_title(f"Accuracy by Prediction Moments ({note}, {vessel})")
            run.log_image(f"{note}-{vessel}", plot=plt)
            for c in columns:
                run.log_list(f"{note}-{vessel}-{c}", temp[c].tolist())


def evaluate(
    config,
    model,
    test_seq,
    le,
    scaler,
    writer=None,
    get_summary=True,
    write_summary=True,
):
    start_time = time.time()
    df_eval = test_seq.df.copy()
    for pred in config.meta["pred_cols"]:
        df_eval[pred] = 0
    if config.with_uncertainty:
        df_eval[
            config.meta["pred_cols"] + ["confidence_pilot", "confidence_berth"]
        ] = predict(config, test_seq, model)
    else:
        df_eval[config.meta["pred_cols"]] = predict(config, test_seq, model)
    df_eval = postprocess(config, df_eval, "test", le, scaler)

    summary = {}
    count_invalid_predictions(config, df_eval, summary)

    if (
        "y_pred_pilot_prod" in df_eval.columns
        and "y_pred_berth_prod" in df_eval.columns
    ):
        logger.info("Comparing with A6 (PROD) predictions")
        baseline_act_cols = ["y_actual_pilot", "y_actual_berth"]
        baseline_pred_cols = ["y_pred_pilot_prod", "y_pred_berth_prod"]
        baseline_is_evaluate_cols = ["is_evaluate_pilot", "is_evaluate_berth"]
        baseline_abs_error_cols = ["abs_error_pilot_prod", "abs_error_berth_prod"]
    else:
        baseline_act_cols = []
        baseline_pred_cols = []
        baseline_is_evaluate_cols = []
        baseline_abs_error_cols = []

    act_cols = config.meta["act_cols"] + baseline_act_cols
    pred_cols = config.meta["pred_cols"] + baseline_pred_cols
    is_evaluate_cols = config.meta["is_evaluate_cols"] + baseline_is_evaluate_cols
    abs_error_cols = config.meta["abs_error_cols"] + baseline_abs_error_cols
    df_eval[abs_error_cols] = np.multiply(
        abs(np.subtract(df_eval[pred_cols], df_eval[act_cols])) / 3600,
        df_eval[is_evaluate_cols],
    )

    df_eval["notes"] = np.where(
        df_eval["is_waiting"] == 1, config.meta["subsets"][1], config.meta["subsets"][2]
    )
    if write_summary:
        df_summary = (
            get_summary_by_prediction_moments(config, df_eval, abs_error_cols)
            if get_summary
            else None
        )
    else:
        df_summary = None

    for subset in config.meta["subsets"]:
        logger.info(
            f"Evaluation - {subset} ------------------------------------------------------------------"
        )
        for abs_error_col in abs_error_cols:
            idc_name = f"{abs_error_col.replace('abs_error', 'mae')} ({subset})"
            error = (
                df_eval[abs_error_col]
                if subset == "overall"
                else df_eval[df_eval["notes"] == subset][abs_error_col]
            )

            logger.info(f"Average {idc_name}: {error.mean()}")
            if run:
                run.log(idc_name, error.mean())
                run.log_row(idc_name, Values=error.mean())
            elif writer:
                summary.update({idc_name: error.mean()})

    if writer and write_summary:
        writer.summary(summary)
        writer_log(df_summary, config, writer)
    time_spent = time.time() - start_time
    config.times["eval_hrs"] = "%d hr %d min" % (
        time_spent // 3600,
        time_spent % 3600 // 60,
    )
    logger.info("Finished Evaluation. Total time spent: %s" % config.times["eval_hrs"])
    return df_eval, df_summary


def export(config, model, le, scaler, writer=None, s3_conn=None):
    tag = model.__class__.__name__
    if config.use_azure:
        output_dir = config.output_path
    elif config.use_allegro:
        output_dir = config.output_path / writer.exp_id
    else:
        output_dir = config.output_path / current_datetime
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    le_path = os.path.join(output_dir, "le.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    model_path = os.path.join(output_dir, tag.lower() + ".pt")
    config_path = os.path.join(output_dir, "config.txt")

    try:
        with open(config_path, "w") as f:
            f.write(json.dumps(config.to_dict(), indent=4))
            f.close()
        logger.info("Configuration exported to %s" % config_path)
        torch.save(model.state_dict(), model_path)
        logger.info("%s model exported to %s" % (tag, model_path))
        save_pickle(le_path, le)
        logger.info("Label encoder exported to %s" % le_path)
        save_pickle(scaler_path, scaler)
        logger.info("Scaler exported to %s" % scaler_path)
    except Exception as e:
        logger.info(f"Fail to save model file to local {e}")

    if config.output_mode == "s3":
        logger.info("Output data to StorageGrid")
        s3_config_path = config_path.replace(GVVMC_HOME + os.path.sep, "")
        s3_le_path = le_path.replace(GVVMC_HOME + os.path.sep, "")
        s3_model_path = model_path.replace(GVVMC_HOME + os.path.sep, "")
        s3_scaler_path = scaler_path.replace(GVVMC_HOME + os.path.sep, "")
        if sys.platform == "win32":
            s3_config_path = s3_config_path.replace(os.path.sep, "/")
            s3_le_path = s3_le_path.replace(os.path.sep, "/")
            s3_model_path = s3_model_path.replace(os.path.sep, "/")
            s3_scaler_path = s3_scaler_path.replace(os.path.sep, "/")
        s3_conn.upload_file(config_path, GVVMC_BUCKET, s3_config_path)
        s3_conn.upload_file(le_path, GVVMC_BUCKET, s3_le_path)
        s3_conn.upload_file(model_path, GVVMC_BUCKET, s3_model_path)
        s3_conn.upload_file(scaler_path, GVVMC_BUCKET, s3_scaler_path)

    if config.use_azure:
        run.upload_file(model_path, model_path)  # Not a bug
        run.register_model("deeprnn", model_path)
        logger.info("Register model finish.")


def export_dashboard_data(config, df_eval, df_summary, exp_id, s3_conn):
    if config.use_azure:
        output_dir = config.output_path
    else:
        output_dir = DATA_DIR / "dashboard_data"
    os.makedirs(output_dir, exist_ok=True)

    filename = output_dir / f"sampled_eval_summary_{exp_id}{config.remarks}.csv"
    df_summary.to_csv(filename, index=False, float_format="%.2f")
    logger.info("Sampled evaluation summary exported to %s" % str(filename))

    if s3_conn is not None and config.output_mode == "s3":
        filename_str = str(filename)
        s3_object_path = filename_str.replace(GVVMC_HOME + os.path.sep, "")
        if sys.platform == "win32":
            s3_object_path = s3_object_path.replace(os.path.sep, "/")
        s3_conn.upload_file(filename_str, GVVMC_BUCKET, s3_object_path)

    df_eval["a6_etp_utc"] = df_eval["time_seen_utc"] + pd.to_timedelta(
        df_eval["y_pred_pilot"], "s"
    )
    df_eval["a6_eta_utc"] = df_eval["time_seen_utc"] + pd.to_timedelta(
        df_eval["y_pred_berth"], "s"
    )
    if "y_pred_pilot_prod" in df_eval.columns:
        df_eval["a6_etp_utc_prod"] = df_eval["time_seen_utc"] + pd.to_timedelta(
            df_eval["y_pred_pilot_prod"], "s"
        )
    if "y_pred_berth_prod" in df_eval.columns:
        df_eval["a6_eta_utc_prod"] = df_eval["time_seen_utc"] + pd.to_timedelta(
            df_eval["y_pred_berth_prod"], "s"
        )
    if "y_pred_pilot_prod_w_seq" in df_eval.columns:
        df_eval["a6_etp_utc_prod_w_seq"] = df_eval["time_seen_utc"] + pd.to_timedelta(
            df_eval["y_pred_pilot_prod_w_seq"], "s"
        )
    if "y_pred_berth_prod_w_seq" in df_eval.columns:
        df_eval["a6_eta_utc_prod_w_seq"] = df_eval["time_seen_utc"] + pd.to_timedelta(
            df_eval["y_pred_berth_prod_w_seq"], "s"
        )
    if "y_pred_lgb_prod" in df_eval.columns:
        df_eval["a5_eta_utc_lgb"] = df_eval["time_seen_utc"] + pd.to_timedelta(
            df_eval["y_pred_lgb_prod"], "s"
        )

    output_cols = [
        "vsl_gid",
        "time_seen_utc",
        "vsl_lat",
        "vsl_lon",
        "speed",
        "heading",
        "teu",
        "vsl_opr_scac",
        "prev_unlocode",
        "dest_unlocode",
        "orig_cs_prt_id",
        "dest_cs_prt_id",
        "orig_tmn_id",
        "dest_tmn_id",
        "orig_tmn_type",
        "dest_tmn_type",
        "orig_atd_utc",
        "dest_ata_utc",
        "atp_ata_utc",
        "dest_day_interval",
        "dest_wkday",
        "dest_country_code",
        "ais_rem_time",
        "haversine_distance",
        "coastal_rem_time",
        "coastal_eta_utc",
        "sch_scac",
        "dest_prt_lat",
        "dest_prt_lon",
        "travel_time",
        "a6_etp_utc",
        "a6_eta_utc",
        "a6_etp_utc_prod",
        "a6_eta_utc_prod",
        "ais_count",
        "abnormal_ais_count",
        "is_evaluate_pilot",
        "is_evaluate_berth",
        "is_valid_route",
        "src",
        "dataset",
        "ais_is_missing",
        "max_diff_in_hour",
        "diff_in_hour",
        "cummax_speed",
        "atp_gt_ata",
        "median_STD_outlier",
        "drifting_far_away",
        "abnormal_speed",
        "is_missing_berth",
        "a6_etp_utc_prod_w_seq",
        "a6_eta_utc_prod_w_seq",
        "ais_eta_utc",
        "y_pred_berth_prod",
        "y_pred_pilot_prod",
        "atp_b4_atd",
        "atp_is_empty",
    ]
    output_cols.extend(
        config.con_features
        + config.cat_features
        + config.seq_features
        + config.meta["pred_cols"]
    )
    if config.with_uncertainty:
        output_cols.extend(["confidence_pilot", "confidence_berth"])
    output_cols = list(
        set(output_cols).intersection(df_eval.columns)
    )  # common columns only
    filename = output_dir / f"eval_{exp_id}{config.remarks}.csv"
    df_eval[output_cols].to_csv(filename, index=False)
    logger.info(
        f"Sampled evaluation data exported to {filename}, size {df_eval.shape[0]}"
    )

    if s3_conn is not None and config.output_mode == "s3":
        filename_str = str(filename)
        s3_object_path = filename_str.replace(GVVMC_HOME + os.path.sep, "")
        if sys.platform == "win32":
            s3_object_path = s3_object_path.replace(os.path.sep, "/")
        s3_conn.upload_file(filename_str, GVVMC_BUCKET, s3_object_path)


def close(config, writer=None):
    config.times["end_time"] = datetime.now()
    total_time_spent = (
        config.times["end_time"] - config.times["init_time"]
    ).total_seconds()
    config.times["total_hrs"] = "%d hr %d min" % (
        total_time_spent // 3600,
        total_time_spent % 3600 // 60,
    )
    logger.info(
        "Finished Execution. Total time spent: %s \n" % config.times["total_hrs"]
    )
    if writer:
        writer.summary(
            {
                k: str(v) if not isinstance(v, (str, int, list)) else v
                for k, v in config.vars.items()
            }
        )
        writer.summary(
            {
                k: str(v) if not isinstance(v, (str, int, list)) else v
                for k, v in config.times.items()
            }
        )


def evaluate_only(exp_id, test_data_path, waiting_time_data_path, remarks=None):
    setup_logger(log_filename=os.path.join(LOG_DIR, "evaluations.log"))
    logger.info("Start Evaluation for experiment %s" % exp_id)
    model_files_dir = MODEL_DIR / exp_id
    config, model, le, scaler = read_historical_model_outputs(model_files_dir)
    config.test_data_path = test_data_path
    config.global_cleansing = True
    config.waiting_time_data_path = waiting_time_data_path
    test_seq = get_testing_data(config, le, scaler)
    df_eval, df_summary = evaluate(config, model, test_seq, le, scaler)
    if remarks:
        config.remarks = remarks
        df_eval["remarks"] = remarks
    export_dashboard_data(config, df_eval, df_summary, exp_id, s3_conn=None)
    close(config)


def construct_split_data_src_path(config, is_waiting_time_model):
    train_df_filename = "train_df"
    val_df_filename = "val_df"
    train_stat_filename = "train_stat"
    test_df_filename = "test_df"

    if is_waiting_time_model:
        waiting_time_append = "_waiting_time"
        train_df_filename += waiting_time_append
        val_df_filename += waiting_time_append
        train_stat_filename += waiting_time_append
        test_df_filename += waiting_time_append

    if config.MODE == "DEBUG":
        debug_append = "_debug"
        train_df_filename += debug_append
        val_df_filename += debug_append
        train_stat_filename += debug_append
        test_df_filename += debug_append

    file_suffix = ".parquet"
    train_stat_filename += ".pkl"
    train_df_filename += file_suffix
    val_df_filename += file_suffix
    test_df_filename += file_suffix

    train_stat_filename = os.path.join(
        DATA_DIR, "a5_raw_data", config.trainset, train_stat_filename
    )
    train_df_filename = os.path.join(
        DATA_DIR, "a5_raw_data", config.trainset, train_df_filename
    )
    val_df_filename = os.path.join(
        DATA_DIR, "a5_raw_data", config.trainset, val_df_filename
    )
    test_df_filename = os.path.join(
        DATA_DIR, "a5_raw_data", config.testset, test_df_filename
    )

    return train_stat_filename, train_df_filename, val_df_filename, test_df_filename


def train_and_evaluate():
    config, writer = setup()
    s3_conn = None

    if args.output_mode == "s3" or args.data_src == "s3":
        logger.info(f"Create connection to StorageGrid with {S3_CONFIG_DIR}")
        with open(S3_CONFIG_DIR, "rb") as f:
            try:
                s3_cred = yaml.safe_load(f)["s3"]
            except yaml.YAMLError as exc:
                print(exc)
        s3_conn = S3Connection(
            end_point=s3_cred["STORAGEGRID_ENDPOINT"],
            access_key=s3_cred["ACCESS_KEY"],
            access_secret=s3_cred["ACCESS_SECRET"],
        )
        s3_conn.connect()

    if args.use_split_data:
        (
            train_stat_path,
            train_df_path,
            val_df_path,
            test_df_path,
        ) = construct_split_data_src_path(
            config, str(args.is_waiting_time_model).lower() in ["true", "1"]
        )

        if args.data_src == "s3":
            for file_path in [
                train_stat_path,
                train_df_path,
                val_df_path,
                test_df_path,
                config.test_data_path,
            ]:
                download_file_from_storage_grid(s3_conn, file_path)

        import pickle

        with open(train_stat_path, "rb") as file:
            train_stat = pickle.load(file)
        le = train_stat["le"]
        scaler = train_stat["scaler"]
        config.vars = train_stat["config"].vars
        config.meta = train_stat["config"].meta

        train_seq = SequenceGenerator(config, path=train_df_path, dataset="train")
        val_seq = SequenceGenerator(config, path=val_df_path, dataset="train")
        test_seq = SequenceGenerator(config, path=test_df_path, dataset="test")
    else:
        train_seq, val_seq, le, scaler = get_training_data(config, s3_conn)
        test_seq = get_testing_data(config, le, scaler, s3_conn)

    if config.enable_hyper_parameter_tuning:
        logger.info("Enable hyper parameter tuning")
        config = hyper_parameter_tuning(
            config=config, train_seq=train_seq, val_seq=val_seq
        )

    model = train(config, train_seq, val_seq, test_seq, ddp=False, writer=writer)
    df_eval, df_summary = evaluate(
        config, model, test_seq, le, scaler, writer, write_summary=True
    )

    if config.PSA or config.keelung or config.fullset_eval:
        df_eval_train, _ = evaluate(
            config, model, train_seq, le, scaler, writer, write_summary=False
        )
        df_eval_val, _ = evaluate(
            config, model, val_seq, le, scaler, writer, write_summary=False
        )
        df_eval = pd.concat(
            [df_eval, df_eval_train, df_eval_val], axis=0, ignore_index=True
        ).reset_index(drop=True)

    export(config, model, le, scaler, writer, s3_conn=s3_conn)
    exp_id = writer.exp_id if writer else current_datetime
    if config.fullset_eval:
        df_eval = df_eval[
            (df_eval.dest_cs_prt_id.isin([21, 39, 425, 426, 429]))
            | (df_eval.dataset == "test")
        ].reset_index(drop=True)
    export_dashboard_data(config, df_eval, df_summary, exp_id, s3_conn=s3_conn)
    close(config, writer)


def ddp_train_and_evaluate(world_size, local_rank, global_rank):
    import pickle

    config, writer = setup(
        world_size=world_size, local_rank=local_rank, global_rank=global_rank
    )
    logger.info(
        f"World Size: {world_size}, Global Rank: {global_rank}, Local Rank: {local_rank}"
    )

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        config.device = torch.device("cuda", local_rank)
    else:
        config.device = torch.device("cuda")
        config.local_rank = 0

    local_rank = local_rank if local_rank != -1 else 0

    (
        train_stat_path,
        train_df_path,
        val_df_path,
        test_df_path,
    ) = construct_split_data_src_path(
        config, str(args.is_waiting_time_model).lower() in ["true", "1"]
    )

    s3_conn = None
    if local_rank == 0 and (args.data_src == "s3" or args.output_mode == "s3"):
        logger.info(f"Create connection to StorageGrid with {S3_CONFIG_DIR}")
        with open(S3_CONFIG_DIR, "rb") as f:
            try:
                s3_cred = yaml.safe_load(f)["s3"]
            except yaml.YAMLError as exc:
                print(exc)
        s3_conn = S3Connection(
            end_point=s3_cred["STORAGEGRID_ENDPOINT"],
            access_key=s3_cred["ACCESS_KEY"],
            access_secret=s3_cred["ACCESS_SECRET"],
        )
        s3_conn.connect()

    if local_rank == 0 and args.data_src == "s3":
        for file_path in [
            train_stat_path,
            train_df_path,
            val_df_path,
            test_df_path,
            config.test_data_path,
        ]:
            download_file_from_storage_grid(s3_conn, file_path)

    dist.barrier()

    with open(train_stat_path, "rb") as file:
        train_stat = pickle.load(file)
    le = train_stat["le"]
    scaler = train_stat["scaler"]
    config.vars = train_stat["config"].vars
    config.meta = train_stat["config"].meta

    train_seq = SequenceGenerator(config, path=train_df_path, dataset="train")
    val_seq = SequenceGenerator(config, path=val_df_path, dataset="train")
    test_seq = SequenceGenerator(config, path=test_df_path, dataset="test")

    # if config.enable_hyper_parameter_tuning:
    #     logger.info("Enable hyper parameter tuning")
    #     config = hyper_parameter_tuning(config=config, train_seq=train_seq, val_seq=val_seq)

    model = train(config, train_seq, val_seq, ddp=True, writer=writer)
    df_eval, df_summary = evaluate(config, model, test_seq, le, scaler, writer)
    if config.PSA or config.keelung or config.fullset_eval:
        df_eval_train, _ = evaluate(config, model, train_seq, le, scaler, writer)
        df_eval_val, _ = evaluate(config, model, val_seq, le, scaler, writer)
        df_eval = pd.concat(
            [df_eval, df_eval_train, df_eval_val], axis=0, ignore_index=True
        ).reset_index(drop=True)

    if global_rank == 0:
        export(config, model, le, scaler, writer, s3_conn=s3_conn)
        exp_id = writer.exp_id if writer else current_datetime
        if config.fullset_eval:
            df_eval = df_eval[
                (df_eval.dest_cs_prt_id.isin([21, 39, 425, 426, 429]))
                | (df_eval.dataset == "test")
            ].reset_index(drop=True)
        export_dashboard_data(config, df_eval, df_summary, exp_id, s3_conn=s3_conn)
        close(config, writer)

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_azure", dest="use_azure")
    parser.add_argument("--mode", dest="mode")
    parser.add_argument("--is_psa", dest="is_psa")
    parser.add_argument("--is_keelung", dest="is_keelung")
    parser.add_argument("--is_waiting_time_model", dest="is_waiting_time_model")
    parser.add_argument("--train_data_path", dest="train_data_path")
    parser.add_argument("--test_data_path", dest="test_data_path")
    parser.add_argument("--valid_num_months", dest="valid_num_months")
    parser.add_argument(
        "--output_mode", dest="output_mode", default="local", choices=["local", "s3"]
    )
    parser.add_argument(
        "--data_src", dest="data_src", default="local", choices=["local", "s3"]
    )
    parser.add_argument("--use_split_data", dest="use_split_data", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    is_ddp = os.environ.get("RANK", -1) != -1
    print(f"DDP: {is_ddp}")

    # run = Run.get_context() if args.use_azure else None
    run = None

    if is_ddp:
        args.use_split_data = True

        backend = "gloo" if sys.platform == "win32" else "nccl"
        dist.init_process_group(backend, timeout=timedelta(minutes=120))

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = args.local_rank
        ddp_train_and_evaluate(world_size, local_rank, global_rank)

        # dist.destroy_process_group()
    else:
        train_and_evaluate()
        # data_path = DATA_DIR / 'a5_raw_data' / '20221213101259' / 'inference_output' / 'test' / 'processed_fullset.parquet'
        # waiting_time_data_path = DATA_DIR / 'a5_raw_data' / '20221213101259' / 'inference_output' / 'port_pair_vsl_date_waiting_stats.csv'
        # evaluate_only('20221213_124218020', data_path, waiting_time_data_path = waiting_time_data_path, remarks='_test_20221213101259')
