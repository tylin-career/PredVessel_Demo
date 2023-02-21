from datetime import datetime
import numpy as np
import os
import torch
from cseta.models.nn.helper import L1Loss, RMSLELoss, MAPELoss


class ModelConfiguration:
    def __init__(self, configs=None):
        self.MODE = ["DEBUG", "FULL_SET"][1]
        self.use_azure = False
        self.use_allegro = True
        self.global_cleansing = True
        self.PSA = False
        self.keelung = False
        self.apply_waiting_time_feature = True
        self.enable_hyper_parameter_tuning = False
        self.is_skip_processing = False
        self.waiting_time_pred_only = False
        self.remarks = ""
        self.with_mae_curve = True
        self.with_test_curve = False
        self.fullset_eval = False

        self.data_folder = "20221213101259"
        self.trainset = os.path.join(self.data_folder, "inference_output", "train")
        self.testset = os.path.join(self.data_folder, "inference_output", "test")
        self.valid_num_months = 1

        # Speed Cleansing
        self.speed_lower_bound = 1
        self.departure_speed_threshold = 6

        self.train_data_path = None
        self.test_data_path = None
        self.output_path = None
        self.log_path = None
        self.file_name = (
            "processed_debug_set.parquet"
            if self.MODE == "DEBUG"
            else "processed_fullset.parquet"
        )
        self.waiting_time_data_path = (
            os.path.join(
                os.environ.get("GVVMC_HOME"),
                "data",
                "a5_raw_data",
                self.data_folder,
                "inference_output",
                "port_pair_vsl_date_waiting_stats.csv",
            )
            if os.environ.get("GVVMC_HOME")
            else None
        )

        self.output_mode = None
        self.data_src = None

        # DDP config
        self.world_size = -1
        self.local_rank = -1
        self.global_rank = -1

        self.sample_by = ["4hrs", "60km"][0]
        self.sample_method = ["random", "latest", "earliest"][0]
        self.target_transform = [
            "None",
            "standard normalization",
            "natural log",
            "boxcox lambda=0.3",
            "log standard normalization",
        ][1]
        self.target = ["y_actual_pilot", "y_actual_berth"]
        self.target_weight = [1, 1]
        self.seq_features = [
            "speed",
            "ais_rem_time",
            "haversine_distance",
            "dlon_sin",
            "dlon_cos",
            "dlat_sin",
            "vsl_lat",
            "vsl_lon",
            "travel_time",
            "is_next_port",
        ]
        self.add_next_port_indicator = "is_next_port" in self.seq_features
        self.con_features = [
            "speed",
            "ais_rem_time",
            "inv_speed",
            "haversine_distance",
            "coastal_rem_time",
            "dest_prt_lat",
            "dest_prt_lon",
            "vsl_lat",
            "vsl_lon",
            "travel_time",
            "time_after_atp",
            "seq_length",
        ]
        self.con_wt_features = ["dest_cos_hour", "dest_sin_hour"]

        self.con_features += self.con_wt_features
        self.con_waiting_time_features = ["prev_n_waiting_hrs_mean"]
        self.con_features += (
            self.con_waiting_time_features if self.apply_waiting_time_feature else []
        )
        self.cat_features = [
            "dest_country_code",
            "sch_scac",
            "dest_cs_prt_id",
            "dest_tmn_id",
            "vsl_opr_id",
            "vsl_size",
            "orig_cs_prt_id",
            "orig_tmn_id",
            "coastal_rem_time_is_nan",
            "ais_rem_time_is_nan",
        ]
        self.cat_wt_features = ["dest_day_interval", "dest_wkday"]
        self.cat_features += self.cat_wt_features

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dual_gpu = False
        self.seed = 1234
        self.checkpoint_ensemble = False
        self.ais_sequence = ["previous ais", "latest ais only", "duplicate latest ais"][
            0
        ]  # switched off PROD AIS cache, using latest only
        self.max_sequence_length = 64
        self.batch_size = 8192 * 2
        self.lr = 0.005
        self.dropout_rate = 0.5
        self.epoch = 20
        self.step_freq = 100
        self.patience = 20
        self.loss_func = [L1Loss, RMSLELoss, MAPELoss][2]
        self.normalized_l1_weight = None  # [[1454.231,6258.352],[1973.87,5316.409],[4441.481,10087.8],[4724.533,9522.821],[8854.741,15772.19],[7457.419,12214.5],[22399.08,32594.59],[20056.79,28598.43],[52866.41,70562.46],[80226.87,102690.1]]
        self.with_uncertainty = False
        self.confidence_alpha = 0.025  # L1Loss: 15000; MAPELoss: 0.025 (suggest to set as 1/10 of the converged loss value without uncertainty)
        self.target_limit = [
            3600,
            42 * 24 * 3600,
        ]  # in MAPELoss, limit the target, unit in seconds
        self.optimizer = torch.optim.Adam
        self.activation = torch.nn.ReLU
        self.batch_normalization = True
        self.lookahead = False
        self.zero_rectifier_slope = False

        self.lstm_num_layer = 1
        self.lstm_hidden_size = 128
        self.conv_out = [8, 16, 32]
        self.conv_kernals = [2, 4, 8, 16]
        self.fc_seq_out = [128, 64]
        self.fc_cat_out = [64, 40]
        self.fc_out = [64, 32, len(self.target)]

        self.meta = {
            "subsets": ["overall", "waiting only", "without waiting only"],
            "act_cols": self.target.copy(),
            "pred_cols": [y.replace("y_actual", "y_pred") for y in self.target],
            "abs_error_cols": [y.replace("y_actual", "abs_error") for y in self.target],
            "is_evaluate_cols": [
                y.replace("y_actual", "is_evaluate") for y in self.target
            ],
            "weight_cols": [y.replace("y_actual", "weight") for y in self.target],
            "interval_cols": [
                "12_hrs",
                "24_hrs",
                "2_days",
                "3_days",
                "5_days",
                "7_days",
                "_14days",
                "_21days",
                "_42days",
                "_42days+",
            ],
            "sort_cols": [
                "vsl_gid",
                "orig_atd_utc",
                "dest_ata_utc",
                "src",
                "time_seen_utc",
            ],  # must include group_cols in the front
            "group_cols": ["vsl_gid", "orig_atd_utc", "dest_ata_utc", "src"],
            "port_pair_cols": ["orig_cs_prt_id", "dest_cs_prt_id"],
        }
        if (
            self.apply_waiting_time_feature
        ):  ## Update meta group cols for waiting time feature
            self.past_n_days_to_cal_waiting = 14
            self.meta.update(
                {
                    "waiting_time_fea_group_col": [
                        "time_seen_date",
                        "prev_unlocode",
                        "dest_unlocode",
                        "is_large_vsl",
                    ]
                }
            )
        self.vars = {
            "embd_size": [],
            "n_class": [],
            "std_norm_const": [],
            "target_mean": None,
            "target_sd": None,
        }
        self.times = {
            "init_time": datetime.now(),
            "end_time": None,
            "preprocess_hrs": None,
            "train_hrs": None,
            "eval_hrs": None,
            "total_hrs": None,
        }
        if configs:
            self.read_dict(configs)
            return

    def to_dict(self, inner_dict=None):
        config_vars = vars(self).copy() if inner_dict is None else inner_dict.copy()
        for k, v in config_vars.items():
            if isinstance(v, dict):
                config_vars.update({k: self.to_dict(inner_dict=v)})
            elif isinstance(v, np.ndarray):
                config_vars.update({k: v.tolist()})
            elif not isinstance(v, (str, int, float, list)):
                config_vars.update({k: str(v)})
        return config_vars

    def read_dict(self, configs: dict):
        class_mapping = {
            "<class 'torch.optim.adam.Adam'>": torch.optim.Adam,
            "Adam": torch.optim.Adam,
            "<class 'torch.nn.modules.loss.L1Loss'>": torch.nn.L1Loss,
            "L1Loss": torch.nn.L1Loss,
            "<class 'torch.nn.modules.loss.MSELoss'>": torch.nn.MSELoss,
            "<class 'cseta.models.nn.helper.MAPELoss'>": MAPELoss,
            "<class 'cseta.models.nn.helper.L1Loss'>": L1Loss,
            "<class 'torch.nn.modules.activation.ReLU'>": torch.nn.ReLU,
            "ReLU": torch.nn.ReLU,
        }
        input_config_keys = configs.keys()
        loaded_config = vars(self)
        added_keys = loaded_config.keys() - input_config_keys
        for key in added_keys:
            loaded_config[key] = False
        for k, v in configs.items():
            if k == "device":
                loaded_config[k] = torch.device(v)
            elif k in ["optimizer", "loss_func", "activation"]:
                loaded_config[k] = class_mapping[v]
            elif k in ["times"]:
                pass
            elif k in loaded_config.keys():
                loaded_config[k] = v
            else:
                loaded_config.update({k: v})


class ModelConfigurationPSA(ModelConfiguration):
    def __init__(self, configs=None):
        super().__init__(configs)
        self.PSA = True

        # overwrite config using PSA model config
        self.MODE = "FULL_SET"
        self.global_cleansing = False
        self.trainset = "20211005131911/inference_output/train"
        self.testset = "20211005131911/inference_output/test"
        self.file_name = (
            "processed_debug_set.pickle"
            if self.MODE == "DEBUG"
            else "processed_fullset.pickle"
        )
        self.valid_num_months = 1

        self.add_next_port_indicator = False
        self.seq_features.remove(
            "is_next_port"
        ) if "is_next_port" in self.seq_features else None
        self.con_features.remove(
            "time_after_atp"
        ) if "time_after_atp" in self.con_features else None
        self.con_features.remove(
            "seq_length"
        ) if "seq_length" in self.con_features else None

        self.ais_sequence = "latest ais only"
        self.max_sequence_length = 1
        self.batch_size = 16
        self.lr = 0.001
        self.epoch = 5
        self.step_freq = 4
        self.patience = 20

        self.fc_cat_out = [32, 16]


class ModelConfigurationKeeLung(ModelConfiguration):
    def __init__(self, configs=None):
        super().__init__(configs)
        self.keelung = True
        # overwrite config using Keelung model config
        self.MODE = "FULL_SET"
        self.global_cleansing = False
        self.trainset = "20211101114526/inference_output/train"
        self.testset = "20211101114526/inference_output/test"
        self.file_name = (
            "processed_debug_set.pickle"
            if self.MODE == "DEBUG"
            else "processed_fullset.pickle"
        )
        self.valid_num_months = 1

        self.add_next_port_indicator = False
        self.seq_features.remove(
            "is_next_port"
        ) if "is_next_port" in self.seq_features else None
        self.con_features.remove(
            "time_after_atp"
        ) if "time_after_atp" in self.con_features else None

        self.ais_sequence = ["previous ais", "latest ais only", "duplicate latest ais"][
            0
        ]  # temp fix to use duplicate latest ais for PROD to avoid AIS cache issue
        self.max_diff_in_hour = 5
        self.continuous_low_speed_times = 3
        self.speed_lower_bound = 4
        self.departure_speed_threshold = 6
        self.batch_size = 1024
        self.lr = 0.005
        self.epoch = 5
        self.step_freq = 4
        self.patience = 20


class WaitingTimeModelConfiguration(ModelConfiguration):
    def __init__(self, configs=None):
        super().__init__(configs)

        self.global_cleansing = True
        self.apply_waiting_time_feature = True
        self.waiting_time_pred_only = False
        self.data_folder = "20220506031713"
        self.waiting_time_data_path = (
            os.path.join(
                os.environ.get("GVVMC_HOME"),
                "data",
                "a5_raw_data",
                self.data_folder,
                "inference_output",
                "port_pair_vsl_date_waiting_stats.csv",
            )
            if os.environ.get("GVVMC_HOME")
            else None
        )

        # overwrite config using waiting time model config
        self.trainset = os.path.join(self.data_folder, "inference_output", "train")
        self.testset = os.path.join(self.data_folder, "inference_output", "test")

        self.batch_size = 8192 * 2
        self.lr = 0.005
        self.dropout_rate = 0.5
        self.epoch = 20
        self.step_freq = 100
        self.patience = 20
        self.loss_func = L1Loss if self.waiting_time_pred_only else MAPELoss
        self.ais_sequence = "previous ais"

        self.con_waiting_time_features = [
            "prev_n_waiting_hrs_mean",
            # "not_berth_waiting_hrs_mean",
        ]
        self.con_features += (
            self.con_waiting_time_features if self.apply_waiting_time_feature else []
        )

        self.target = (
            ["y_actual_berth"]
            if self.waiting_time_pred_only
            else ["y_actual_pilot", "y_actual_berth"]
        )

        self.fc_out = [64, 32, len(self.target)]
        self.meta.update(
            {
                "waiting_time_fea_group_col": [
                    "time_seen_date",
                    "prev_unlocode",
                    "dest_unlocode",
                    "is_large_vsl",
                ],
                "act_cols": self.target.copy(),
                "pred_cols": [y.replace("y_actual", "y_pred") for y in self.target],
                "abs_error_cols": [
                    y.replace("y_actual", "abs_error") for y in self.target
                ],
                "is_evaluate_cols": [
                    y.replace("y_actual", "is_evaluate") for y in self.target
                ],
                "weight_cols": [y.replace("y_actual", "weight") for y in self.target],
            }
        )
        self.past_n_days_to_cal_waiting = 14
