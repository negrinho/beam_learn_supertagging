#%%

import research_toolbox.tb_io as tb_io
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_utils as tb_ut

# %% table 1: trained without beam aware models. solved with vanilla beam search.
i = 1000
for m in ["vaswani", "lm"]:
    for z in ["supertagging"]:
        for x in [1]:
            for y in ["continue", "reset", "stop"]:
                tb_io.write_jsonfile(
                    {
                        "_overlays_": ["configs/cfgref.json"],
                        "model_type": m,
                        "data_type": z,
                        "beam_size": x,
                        "traj_type": y,
                        "out_folder": "out/cfg%d" % i
                    }, "configs/cfg%d.json" % i)
                i += 1

# %% table 2: trained with beam aware models and multiple data collection strategies. ran the same way.

i = 2000
for m in ["vaswani", "lm"]:
    for z in ["supertagging"]:
        for x in [1, 2, 4, 8]:
            for y in ["oracle", "continue", "reset", "reset_multiple", "stop"]:
                tb_io.write_jsonfile(
                    {
                        "_overlays_": ["configs/cfgref.json"],
                        "model_type": m,
                        "data_type": z,
                        "beam_size": x,
                        "traj_type": y,
                        "out_folder": "out/cfg%d" % i
                    }, "configs/cfg%d.json" % i)
                i += 1

# %% table 3: trained with beam aware models and multiple loss functions. ran the same way.

i = 3000
for m in ["vaswani", "lm"]:
    for z in ["supertagging"]:
        for x in [1, 2, 4, 8]:
            for y in ["continue"]:
                for w in [
                        "log_neighbors", "log_beam",
                        "cost_sensitive_margin_last", "margin_last",
                        "perceptron_first", "perceptron_last", "upper_bound"
                ]:
                    tb_io.write_jsonfile(
                        {
                            "_overlays_": ["configs/cfgref.json"],
                            "model_type": m,
                            "data_type": z,
                            "beam_size": x,
                            "traj_type": y,
                            "loss_type": w,
                            "out_folder": "out/cfg%d" % i
                        }, "configs/cfg%d.json" % i)
                    i += 1

# %% score accumulation experiments

# the experiments with score accumulation should be glanced from the other table.
i = 4000
for m in ["vaswani", "lm"]:
    for v in [0, 1]:
        for z in ["supertagging"]:
            for x in [1, 2, 4, 8]:
                for y in ["continue"]:
                    for w in [
                            "log_neighbors", "log_beam",
                            "cost_sensitive_margin_last"
                    ]:
                        tb_io.write_jsonfile(
                            {
                                "_overlays_": ["configs/cfgref.json"],
                                "model_type": m,
                                "data_type": z,
                                "beam_size": x,
                                "traj_type": y,
                                "loss_type": w,
                                "accumulate_scores": v,
                                "out_folder": "out/cfg%d" % i
                            }, "configs/cfg%d.json" % i)
                        i += 1

# %% update only on cost increase experiments

i = 5000
for m in ["vaswani", "lm"]:
    for z in ["supertagging"]:
        for v in [0, 1]:
            for x in [1, 2, 4, 8]:
                for y in ["continue"]:
                    for w in [
                            "log_neighbors", "log_beam",
                            "cost_sensitive_margin_last"
                    ]:
                        tb_io.write_jsonfile(
                            {
                                "_overlays_": ["configs/cfgref.json"],
                                "model_type": m,
                                "data_type": z,
                                "beam_size": x,
                                "traj_type": y,
                                "update_only_on_cost_increase": v,
                                "loss_type": w,
                                "out_folder": "out/cfg%d" % i
                            }, "configs/cfg%d.json" % i)
                        i += 1

# %% check that training for longer does not result in substantially different results.
i = 6000
for m in ["vaswani", "lm"]:
    for z in ["supertagging"]:
        for x in [1, 2, 4, 8]:
            for y in ["oracle", "continue", "reset", "reset_multiple", "stop"]:
                tb_io.write_jsonfile(
                    {
                        "_overlays_": ["configs/cfgref.json"],
                        "num_epochs": 32,
                        "model_type": m,
                        "data_type": z,
                        "beam_size": x,
                        "traj_type": y,
                        "out_folder": "out/cfg%d" % i
                    }, "configs/cfg%d.json" % i)
                i += 1

# %% Process the existing configs to run for some number of configs.

import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_io as tb_io

num_repeats = 3
all_configs = tb_fs.list_files('configs')

exp_configs = [x for x in all_configs if not x.endswith('cfgref.json')]
for x in exp_configs:
    start_idx = len('configs/cfg')
    i = int(x[start_idx:-5])
    # print(i)
    for r in range(num_repeats):
        tb_io.write_jsonfile(
            {
                "_overlays_": [x],
                'out_folder': "out/cfg_r%d_%d" % (r, i)
            }, "configs/cfg_r%d_%d.json" % (r, i))

# %%
