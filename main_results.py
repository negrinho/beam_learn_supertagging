#%%
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import research_toolbox.tb_io as tb_io
import seaborn as sns
sns.set(color_codes=True)

num_repeats = 3
show_plot = True


def show_plots(xs,
               key,
               label_fn=None,
               xlabel=None,
               ylabel=None,
               show_legend=True,
               ref_line=None,
               filepath=None):

    for x in xs:
        print(x)
        y_lst = []
        for i in range(num_repeats):
            r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
            c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                                  (i, x))

            # assumes that for the repeats is always the same,
            if label_fn is not None:
                label = label_fn(c, r, x)
            else:
                label = str(x)

            y_lst.append(r[key])

        ys = np.stack(y_lst)
        # plt.plot(np.arange(1, len(r[key]) + 1), np.mean(ys, axis=0), label=label)
        plt.errorbar(np.arange(1,
                               len(r[key]) + 1),
                     np.mean(ys, axis=0),
                     yerr=np.std(ys, axis=0),
                     label=label)

    if ref_line is not None:
        plt.axhline(ref_line, color='k', linestyle='--')

    # labelling the plot
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_legend:
        plt.legend()

    if filepath != None:
        plt.savefig(filepath, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def read_config(n):
    return tb_io.read_jsonfile_with_overlays("configs/cfg%s.json" % n)


def read_results(n):
    return tb_io.read_jsonfile("out/cfg%s/checkpoint.json" % n)


# %% for the section 1 (training without beam-aware losses)
# NOTE: the plots pertaining to the losses are commented out as they are not included in the paper.


def label_fn(config, results, idx):
    return config["traj_type"]


# vaswani
xs = [1000, 1001, 1002]
show_plots(xs,
           "dev_acc",
           label_fn=label_fn,
           xlabel="Epoch",
           ylabel="Validation accuracy",
           filepath="figures_paper/table1/dev_acc_vaswani.pdf")
show_plots(xs,
           "train_acc",
           label_fn=label_fn,
           show_legend=False,
           xlabel="Epoch",
           ylabel="Training accuracy",
           filepath="figures_paper/table1/train_acc_vaswani.pdf")
# show_plots(xs,
#            "avg_loss",
#            label_fn=label_fn,
#            show_legend=False,
#            xlabel="Epoch",
#            ylabel="Training loss",
#            filepath="figures_paper/table1/avg_loss_vaswani.pdf")

# lm
xs = [1003, 1004, 1005]
show_plots(xs,
           "dev_acc",
           label_fn=label_fn,
           xlabel="Epoch",
           ylabel="Validation accuracy",
           filepath="figures_paper/table1/dev_acc_lm.pdf")
show_plots(xs,
           "train_acc",
           label_fn=label_fn,
           show_legend=False,
           xlabel="Epoch",
           ylabel="Training accuracy",
           filepath="figures_paper/table1/train_acc_lm.pdf")
# show_plots(xs,
#            "avg_loss",
#            label_fn=label_fn,
#            show_legend=False,
#            xlabel="Epoch",
#            ylabel="Training loss",
#            filepath="figures_paper/table1/avg_loss_lm.pdf")

# # %% for the section 2 (table with the beam size and data collection strategy.)


def label_fn(config, results, idx):
    return "%s %d" % (config["traj_type"], config["beam_size"])


# vaswani
xs = [2001, 2002, 2006, 2007, 2011, 2012]
show_plots(xs,
           "dev_acc",
           label_fn=label_fn,
           xlabel="Epoch",
           ylabel="Validation accuracy",
           filepath="figures_paper/table2/dev_acc_vaswani.pdf")
show_plots(xs,
           "train_acc",
           label_fn=label_fn,
           show_legend=False,
           xlabel="Epoch",
           ylabel="Training accuracy",
           filepath="figures_paper/table2/train_acc_vaswani.pdf")
# show_plots(xs,
#            "avg_loss",
#            label_fn=label_fn,
#            show_legend=False,
#            xlabel="Epoch",
#            ylabel="Training loss",
#            filepath="figures_paper/table2/avg_loss_vaswani.pdf")

# lm
xs = [2021, 2022, 2026, 2027, 2031, 2032]
show_plots(xs,
           "dev_acc",
           label_fn=label_fn,
           xlabel="Epoch",
           ylabel="Validation accuracy",
           filepath="figures_paper/table2/dev_acc_lm.pdf")
show_plots(xs,
           "train_acc",
           label_fn=label_fn,
           show_legend=False,
           xlabel="Epoch",
           ylabel="Training accuracy",
           filepath="figures_paper/table2/train_acc_lm.pdf")
# show_plots(xs,
#            "avg_loss",
#            label_fn=label_fn,
#            show_legend=False,
#            xlabel="Epoch",
#            ylabel="Training loss",
#            filepath="figures_paper/table2/avg_loss_lm.pdf")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %% These results were obtained by running main.py with the compute_vanilla_beam_accuracy
# flag for the appropriate configs.

# cfg_r0_1000.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r0_1000',
#  u'traj_type': u'continue',
# (1, 0.9410197701554313, 0.9419732350869588)
# (2, 0.9410197701554313, 0.9421718950353073)
# (4, 0.9410858174452909, 0.9420996550540897)
# (8, 0.9410858174452909, 0.9420996550540897)

# cfg_r0_1001.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r0_1001',
#  u'traj_type': u'reset',
# (1, 0.9383558627977632, 0.9418106951292192)
# (2, 0.9389282726432125, 0.9417023351573929)
# (4, 0.9389502884064991, 0.9417745751386105)
# (8, 0.9389502884064991, 0.9418648751151325)

# cfg_r0_1002.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r0_1002',
#  u'traj_type': u'stop',
# (1, 0.9374972480295892, 0.9398421556410396)
# (2, 0.9380256263484655, 0.9404200754907803)
# (4, 0.938091673638325, 0.9404381354860848)
# (8, 0.9381797366914711, 0.9404381354860848)

# cfg_r0_1003.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r0_1003',
#  u'traj_type': u'continue',
# (1, 0.8193166307075866, 0.8235899658666088)
# (2, 0.8226630267271366, 0.8276354048147947)
# (4, 0.8225529479107041, 0.8291705044156689)
# (8, 0.8230372947030073, 0.8291163244297557)

# cfg_r0_1004.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r0_1004',
#  u'traj_type': u'reset',
# (1, 0.7332570120206068, 0.7259576312510159)
# (2, 0.7668310510325393, 0.7607231222119882)
# (4, 0.7764519395887456, 0.7737985588123747)
# (8, 0.7774206331733521, 0.7754420183850752)

# cfg_r0_1005.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r0_1005',
#  u'traj_type': u'stop',
# (1, 0.7449693980890317, 0.7469433457947301)
# (2, 0.7720267711681564, 0.7753155984179444)
# (4, 0.7789617366034081, 0.7824312365678785)
# (8, 0.7794240676324248, 0.7842914160842318)

# cfg_r1_1000.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r1_1000',
#  u'traj_type': u'continue',
# (1, 0.9406455021795606, 0.9418829351104369)
# (2, 0.9409537228655718, 0.9420093550775677)
# (4, 0.9409317071022852, 0.9420996550540897)
# (8, 0.9409757386288583, 0.9420996550540897)

# cfg_r1_1001.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r1_1001',
#  u'traj_type': u'reset',
# (1, 0.9388402095900665, 0.940943815354608)
# (2, 0.938862225353353, 0.9417926351339149)
# (4, 0.93881819382678, 0.9419190551010457)
# (8, 0.93881819382678, 0.9419190551010457)

# cfg_r1_1002.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r1_1002',
#  u'traj_type': u'stop',
# (1, 0.9385760204306284, 0.9391378158241679)
# (2, 0.9392585090925102, 0.9400408155893879)
# (4, 0.9391704460393642, 0.9399685756081704)
# (8, 0.9391704460393642, 0.9399685756081704)

# cfg_r1_1003.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r1_1003',
#  u'traj_type': u'continue',
# (1, 0.8203073400554797, 0.8249805855050477)
# (2, 0.8232134208092995, 0.8272742049087067)
# (4, 0.824005988287614, 0.8280507847067959)
# (8, 0.8242481616837656, 0.8282133246645356)

# cfg_r1_1004.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r1_1004',
#  u'traj_type': u'reset',
# (1, 0.7276870239091189, 0.7273663108847591)
# (2, 0.7621196776892255, 0.7610120821368587)
# (4, 0.7703535731583814, 0.7722634592115006)
# (8, 0.7723790233807406, 0.7739069187842011)

# cfg_r1_1005.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r1_1005',
#  u'traj_type': u'stop',
# (1, 0.7402800405090044, 0.745859746076466)
# (2, 0.7687684382017524, 0.7732206389626338)
# (4, 0.7756593721104311, 0.7800111971970888)
# (8, 0.7775086962264982, 0.7816365967744848)

# cfg_r2_1000.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r2_1000',
#  u'traj_type': u'continue',
# (1, 0.9394566509620889, 0.9415759151902621)
# (2, 0.9395667297785214, 0.9415036752090444)
# (4, 0.9395887455418079, 0.9415397951996533)
# (8, 0.9395887455418079, 0.9415397951996533)

# cfg_r2_1001.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r2_1001',
#  u'traj_type': u'reset',
# (1, 0.9360882391792523, 0.9402936555236495)
# (2, 0.9366386332614152, 0.9409799353452168)
# (4, 0.9367046805512748, 0.9409799353452168)
# (8, 0.9368147593677073, 0.9409799353452168)

# cfg_r2_1002.json
#  u'model_type': u'vaswani',
#  u'out_folder': u'out/cfg_r2_1002',
#  u'traj_type': u'stop',
# (1, 0.9397648716481001, 0.9419009951057413)
# (2, 0.9397648716481001, 0.9424247349695689)
# (4, 0.9397868874113866, 0.9423524949883513)
# (8, 0.939808903174673, 0.9423705549836557)

# cfg_r2_1003.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r2_1003',
#  u'traj_type': u'continue',
# (1, 0.8199991193694686, 0.8234093659135648)
# (2, 0.8232134208092995, 0.8261544851998338)
# (4, 0.8244022720267712, 0.8268227050260967)
# (8, 0.8250187133987935, 0.8275812248288815)

# cfg_r2_1004.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r2_1004',
#  u'traj_type': u'reset',
# (1, 0.7349742415569548, 0.7311227899080747)
# (2, 0.7675135396944212, 0.7619873218832963)
# (4, 0.7756593721104311, 0.7725885391269798)
# (8, 0.7763418607723129, 0.7748640985353343)

# cfg_r2_1005.json
#  u'model_type': u'lm',
#  u'out_folder': u'out/cfg_r2_1005',
#  u'traj_type': u'stop',
# (1, 0.7451675399586104, 0.7450831662783768)
# (2, 0.7710140460569768, 0.7723356991927182)
# (4, 0.7773545858834926, 0.7796319372956964)
# (8, 0.7775527277530712, 0.7806432970327428)

# %%%

d = {}

d[('vaswani', 'continue', 0)] = [(1, 0.9410197701554313, 0.9419732350869588),
                                 (2, 0.9410197701554313, 0.9421718950353073),
                                 (4, 0.9410858174452909, 0.9420996550540897),
                                 (8, 0.9410858174452909, 0.9420996550540897)]

d[('vaswani', 'reset', 0)] = [(1, 0.9383558627977632, 0.9418106951292192),
                              (2, 0.9389282726432125, 0.9417023351573929),
                              (4, 0.9389502884064991, 0.9417745751386105),
                              (8, 0.9389502884064991, 0.9418648751151325)]

d[('vaswani', 'stop', 0)] = [(1, 0.9374972480295892, 0.9398421556410396),
                             (2, 0.9380256263484655, 0.9404200754907803),
                             (4, 0.938091673638325, 0.9404381354860848),
                             (8, 0.9381797366914711, 0.9404381354860848)]

d[('lm', 'continue', 0)] = [(1, 0.8193166307075866, 0.8235899658666088),
                            (2, 0.8226630267271366, 0.8276354048147947),
                            (4, 0.8225529479107041, 0.8291705044156689),
                            (8, 0.8230372947030073, 0.8291163244297557)]

d[('lm', 'reset', 0)] = [(1, 0.7332570120206068, 0.7259576312510159),
                         (2, 0.7668310510325393, 0.7607231222119882),
                         (4, 0.7764519395887456, 0.7737985588123747),
                         (8, 0.7774206331733521, 0.7754420183850752)]

d[('lm', 'stop', 0)] = [(1, 0.7449693980890317, 0.7469433457947301),
                        (2, 0.7720267711681564, 0.7753155984179444),
                        (4, 0.7789617366034081, 0.7824312365678785),
                        (8, 0.7794240676324248, 0.7842914160842318)]

d[('vaswani', 'continue', 1)] = [(1, 0.9406455021795606, 0.9418829351104369),
                                 (2, 0.9409537228655718, 0.9420093550775677),
                                 (4, 0.9409317071022852, 0.9420996550540897),
                                 (8, 0.9409757386288583, 0.9420996550540897)]

d[('vaswani', 'reset', 1)] = [(1, 0.9388402095900665, 0.940943815354608),
                              (2, 0.938862225353353, 0.9417926351339149),
                              (4, 0.93881819382678, 0.9419190551010457),
                              (8, 0.93881819382678, 0.9419190551010457)]

d[('vaswani', 'stop', 1)] = [(1, 0.9385760204306284, 0.9391378158241679),
                             (2, 0.9392585090925102, 0.9400408155893879),
                             (4, 0.9391704460393642, 0.9399685756081704),
                             (8, 0.9391704460393642, 0.9399685756081704)]

d[('lm', 'continue', 1)] = [(1, 0.8203073400554797, 0.8249805855050477),
                            (2, 0.8232134208092995, 0.8272742049087067),
                            (4, 0.824005988287614, 0.8280507847067959),
                            (8, 0.8242481616837656, 0.8282133246645356)]

d[('lm', 'reset', 1)] = [(1, 0.7276870239091189, 0.7273663108847591),
                         (2, 0.7621196776892255, 0.7610120821368587),
                         (4, 0.7703535731583814, 0.7722634592115006),
                         (8, 0.7723790233807406, 0.7739069187842011)]

d[('lm', 'stop', 1)] = [(1, 0.7402800405090044, 0.745859746076466),
                        (2, 0.7687684382017524, 0.7732206389626338),
                        (4, 0.7756593721104311, 0.7800111971970888),
                        (8, 0.7775086962264982, 0.7816365967744848)]

d[('vaswani', 'continue', 2)] = [(1, 0.9394566509620889, 0.9415759151902621),
                                 (2, 0.9395667297785214, 0.9415036752090444),
                                 (4, 0.9395887455418079, 0.9415397951996533),
                                 (8, 0.9395887455418079, 0.9415397951996533)]

d[('vaswani', 'reset', 2)] = [(1, 0.9360882391792523, 0.9402936555236495),
                              (2, 0.9366386332614152, 0.9409799353452168),
                              (4, 0.9367046805512748, 0.9409799353452168),
                              (8, 0.9368147593677073, 0.9409799353452168)]

d[('vaswani', 'stop', 2)] = [(1, 0.9397648716481001, 0.9419009951057413),
                             (2, 0.9397648716481001, 0.9424247349695689),
                             (4, 0.9397868874113866, 0.9423524949883513),
                             (8, 0.939808903174673, 0.9423705549836557)]

d[('lm', 'continue', 2)] = [(1, 0.8199991193694686, 0.8234093659135648),
                            (2, 0.8232134208092995, 0.8261544851998338),
                            (4, 0.8244022720267712, 0.8268227050260967),
                            (8, 0.8250187133987935, 0.8275812248288815)]

d[('lm', 'reset', 2)] = [(1, 0.7349742415569548, 0.7311227899080747),
                         (2, 0.7675135396944212, 0.7619873218832963),
                         (4, 0.7756593721104311, 0.7725885391269798),
                         (8, 0.7763418607723129, 0.7748640985353343)]

d[('lm', 'stop', 2)] = [(1, 0.7451675399586104, 0.7450831662783768),
                        (2, 0.7710140460569768, 0.7723356991927182),
                        (4, 0.7773545858834926, 0.7796319372956964),
                        (8, 0.7775527277530712, 0.7806432970327428)]

k2r = {}
for k, vs in d.items():
    for v in vs:
        out_k = k[:2] + tuple(v[:1])
        if out_k not in k2r:
            k2r[out_k] = []
        k2r[out_k].append(v[1])

pprint(out_d)

for k1 in ["vaswani", "lm"]:
    print
    for k2 in ["reset", "continue", "stop"]:
        print " & ".join([
            "$%0.2f_{%0.2f}$" % (round(100.0 * np.mean(k2r[(k1, k2, k3)]), 2),
                                 round(100.0 * np.std(k2r[(k1, k2, k3)]), 2))
            for k3 in [1, 2, 4, 8]
        ])

# %%

k2r = {}
for i in range(num_repeats):
    for x in range(2000, 2040):
        r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
        c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                              (i, x))
        k = (c["model_type"], c["traj_type"], c['beam_size'])

        if k not in k2r:
            k2r[k] = []
        k2r[k].append(np.max(r["dev_acc"]))

# generate the table.
for k1 in ["vaswani", "lm"]:
    print
    for k2 in ["oracle", "reset", "reset_multiple", "continue", "stop"]:
        print " & ".join([
            "$%0.2f_{%0.2f}$" % (round(100.0 * np.mean(k2r[(k1, k2, k3)]), 2),
                                 round(100.0 * np.std(k2r[(k1, k2, k3)]), 2))
            for k3 in [1, 2, 4, 8]
        ])

# %%

# "model_type": m,
# "data_type": z,
# "beam_size": x,
# "traj_type": y,
# "loss_type": w,

k2r = {}
for i in range(num_repeats):
    for x in range(3000, 3056):
        r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
        c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                              (i, x))
        k = (c["model_type"], c["loss_type"], c['beam_size'])
        if k not in k2r:
            k2r[k] = []
        k2r[k].append(np.max(r["dev_acc"]))

# generate the table.
for k1 in ["vaswani", "lm"]:
    print
    for k2 in [
            "perceptron_first", "perceptron_last", "margin_last",
            "cost_sensitive_margin_last", "log_beam", "log_neighbors"
    ]:
        print " & ".join([
            "$%0.2f_{%0.2f}$" % (round(100.0 * np.mean(k2r[(k1, k2, k3)]), 2),
                                 round(100.0 * np.std(k2r[(k1, k2, k3)]), 2))
            for k3 in [1, 2, 4, 8]
        ])

# %% score accumulation versus non score accumulation.

k2r = {}
for i in range(num_repeats):
    for x in range(4000, 4048):
        r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
        c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                              (i, x))
        k = (c["model_type"], c["loss_type"], c["accumulate_scores"],
             c['beam_size'])
        if k not in k2r:
            k2r[k] = []
        k2r[k].append(np.max(r["dev_acc"]))

# # generate the table.
for k1 in ["vaswani", "lm"]:
    print
    for k2 in [0, 1]:
        print
        print
        for k3 in ["log_neighbors", "log_beam", "cost_sensitive_margin_last"]:
            print " & ".join([
                "$%0.2f_{%0.2f}$" %
                (round(100.0 * np.mean(k2r[(k1, k3, k2, k4)]),
                       2), round(100.0 * np.std(k2r[(k1, k3, k2, k4)]), 2))
                for k4 in [1, 2, 4, 8]
            ])

# %% update only on cost increase

k2r = {}
k2n = {}
for i in range(num_repeats):
    for x in range(5000, 5048):
        r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
        c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                              (i, x))
        k = (c["model_type"], c["loss_type"], c["update_only_on_cost_increase"],
             c['beam_size'])
        if k not in k2r:
            k2r[k] = []
        k2r[k].append(np.max(r["dev_acc"]))
pprint(k2r.keys())

# generate the table.
for k1 in ["vaswani", "lm"]:
    print
    for k2 in [0, 1]:
        print
        print
        for k3 in ["log_neighbors", "log_beam"]:
            print " & ".join([
                "$%0.2f_{%0.2f}$" %
                (round(100.0 * np.mean(k2r[(k1, k3, k2, k4)]),
                       2), round(100.0 * np.std(k2r[(k1, k3, k2, k4)]), 2))
                for k4 in [1, 2, 4, 8]
            ])

# %% train for double the number of epochs to guarantee that the performance is approximately the same.

k2r = {}
for i in range(num_repeats):
    for x in range(6000, 6040):
        r = tb_io.read_jsonfile("out/cfg_r%d_%d/checkpoint.json" % (i, x))
        c = tb_io.read_jsonfile_with_overlays("configs/cfg_r%d_%d.json" %
                                              (i, x))
        k = (c["model_type"], c["traj_type"], c['beam_size'])
        if k not in k2r:
            k2r[k] = []
        k2r[k].append(np.max(r["dev_acc"]))

# generate the table.
for k1 in ["vaswani", "lm"]:
    print
    for k2 in ["oracle", "reset", "reset_multiple", "continue", "stop"]:
        print " & ".join([
            "$%0.2f_{%0.2f}$" % (round(100.0 * np.mean(k2r[(k1, k2, k3)]), 2),
                                 round(100.0 * np.std(k2r[(k1, k2, k3)]), 2))
            for k3 in [1, 2, 4, 8]
        ])
