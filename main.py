#%%
# import dynet_config
# dynet_config.set(random_seed=0)
# TODO: set the random seed for numpy and dynet
import dynet as dy
import numpy as np
import random
from pprint import pprint
from itertools import izip
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_experiments as tb_ex
import research_toolbox.tb_io as tb_io
import sys


def cosine_get_lr(rate_start, rate_end, duration, idx, logspace=False):
    if logspace:
        rate_start = math.log10(rate_start)
        rate_end = math.log10(rate_end)

    lr = rate_end + 0.5 * (rate_start - rate_end) * (
        1 + np.cos(float(idx) / (duration - 1) * np.pi))

    if logspace:
        lr = 10.0**lr

    return lr


def get_beams(backpointers):
    beams = []
    beam_prev = [[]]
    for i, (beam_indices, tag_indices) in enumerate(backpointers):
        beam_cur = [
            beam_prev[b_idx] + [t_idx]
            for b_idx, t_idx in izip(beam_indices, tag_indices)
        ]

        beams.append(beam_cur)
        beam_prev = beam_cur
    return beams


# def print_all_expr_dims(d):
#     pprint({
#         k: v.dim()[0] for k, v in d.iteritems() if isinstance(v, dy.Expression)
#     })

# def print_all_expr_values(d):
#     out_d = {
#         k: v.npvalue()
#         for k, v in d.iteritems()
#         if isinstance(v, dy.Expression)
#     }
#     print_all_np_values(out_d)

# def print_all_np_dims(d):
#     pprint({k: v.shape for k, v in d.iteritems() if type(v) == np.ndarray})

# def print_all_np_values(d):
#     out_d = {}
#     for k, v in d.iteritems():
#         if type(v) == np.ndarray:
#             v = np.round(v, 2)
#             if len(v.shape) == 1:
#                 out_d[k] = v[:4]
#             elif len(v.shape) == 2:
#                 if k != "scores":
#                     out_d[k] = v[:4, :4]
#                 else:
#                     out_d[k] = v
#             else:
#                 raise ValueError
#     for k, v in out_d.iteritems():
#         print k
#         print v

########### INITIALIZATION ###########

idx = sys.argv.index('--config_filepath') + 1
cfg_filepath = sys.argv[idx]
cfg = tb_io.read_jsonfile_with_overlays(cfg_filepath)
pprint(cfg)

tb_fs.create_folder(cfg["out_folder"],
                    abort_if_exists=False,
                    create_parent_folders=True)

if '--train' in sys.argv:
    tb_io.write_jsonfile(cfg, cfg["out_folder"] + "/cfg.json")

if cfg["data_type"] == "supertagging":
    tags_key = "supertags"
    train_data = tb_io.read_jsonlogfile('data/supertagging/train.jsonl')
    dev_data = tb_io.read_jsonlogfile('data/supertagging/dev.jsonl')
    test_data = tb_io.read_jsonlogfile('data/supertagging/test.jsonl')
# elif cfg["data_type"] == "conll2000":
#     tags_key = "chunk_tags"
#     test_data = tb_io.read_jsonlogfile('data/conll2000/test.jsonl')
#     train_data = tb_io.read_jsonlogfile('data/conll2000/train.jsonl')
#     n = len(train_data)
#     num_dev = int(0.2 * n)
#     dev_data = train_data[:num_dev]
#     train_data = train_data[num_dev:]
# elif cfg["data_type"] == "conll2003":
#     tags_key = "ner_tags"
#     train_data = tb_io.read_jsonlogfile('data/conll2003/train.jsonl')
#     dev_data = tb_io.read_jsonlogfile('data/conll2003/dev.jsonl')
#     test_data = tb_io.read_jsonlogfile('data/conll2003/test.jsonl')
# elif cfg["data_type"] == "ptb":
#     assert not cfg["use_postags"]
#     tags_key = "postags"
#     train_data = tb_io.read_jsonlogfile('data/ptb/train.jsonl')
#     dev_data = tb_io.read_jsonlogfile('data/ptb/dev.jsonl')
#     test_data = tb_io.read_jsonlogfile('data/ptb/test.jsonl')
else:
    raise ValueError

if cfg["debug"]:
    train_data = train_data[:cfg["num_debug"]]
    dev_data = dev_data[:cfg["num_debug"]]
    test_data = test_data[:cfg["num_debug"]]

import research_toolbox.tb_preprocessing as tb_pr
tk_tags_key = 'tk_' + tags_key
w2cnt = tb_pr.count_tokens([e["words"] for e in train_data])
all_words = tb_pr.remove_rare_tokens(w2cnt, 2)
w2idx = tb_pr.index_tokens(all_words, 1)
w2idx["_UNK_"] = 0
num_words = len(w2idx)

# # substitute letter by a (or A) depending on capitalization
# # substitute number by 0
# def compute_word_shape(w):
#     lst = []
#     for ch in w:
#         if ch >= "a" and ch <= "z":
#             lst.append("a")
#         elif ch >= "A" and ch <= "Z":
#             lst.append("A")
#         elif ch >= "0" and ch <= "9":
#             lst.append("0")
#         else:
#             lst.append(ch)
#     s = "".join(lst)
#     return s

# if cfg["use_word_shapes"]:
#     shape2cnt = {}
#     for w, c in w2cnt.iteritems():
#         lst = []
#         for ch in w:
#             if ch >= "a" and ch <= "z":
#                 lst.append("a")
#             elif ch >= "A" and ch <= "Z":
#                 lst.append("A")
#             elif ch >= "0" and ch <= "9":
#                 lst.append("0")
#             else:
#                 lst.append(ch)
#         s = "".join(lst)
#         if s not in shape2cnt:
#             shape2cnt[s] = 0
#         shape2cnt[s] += c
#     all_shapes = tb_pr.remove_rare_tokens(shape2cnt, 16)
#     shape2idx = tb_pr.index_tokens(all_shapes, 1)
#     shape2idx["_UNK_"] = 0
#     num_shapes = len(shape2idx)
# NOTE: 32 is a good threshold.
# sh2e
# use_shapes
# sh_emb_dim

if cfg["use_postags"]:
    pos2cnt = tb_pr.count_tokens([e["postags"] for e in train_data])
    all_postags = tb_pr.remove_rare_tokens(pos2cnt, 2)
    pos2idx = tb_pr.index_tokens(all_postags, 1)
    pos2idx["_UNK_"] = 0
    num_pos = len(pos2idx)

t2cnt = tb_pr.count_tokens([e[tags_key] for e in train_data])
t2idx = tb_pr.index_tokens(t2cnt.keys())
idx2t = {v: k for (k, v) in t2idx.iteritems()}
num_tags = len(t2idx)
# print len(w2idx), len(t2idx)

# NOTE: this gives the correct number of tags.
# print len({t: c for t, c in t2cnt.iteritems() if c >= 10})

if '--train' in sys.argv:
    tb_io.write_jsonfile(w2idx, cfg["out_folder"] + "/word2idx.json")
    if cfg["use_postags"]:
        tb_io.write_jsonfile(pos2idx, cfg["out_folder"] + "/pos2idx.json")
    tb_io.write_jsonfile(t2idx, cfg["out_folder"] + "/tag2idx.json")

for e in train_data:
    e["tk_words"] = [w2idx.get(w, 0) for w in e["words"]]
    e[tk_tags_key] = [t2idx[t] for t in e[tags_key]]
    if cfg["use_postags"]:
        e["tk_postags"] = [pos2idx.get(t, 0) for t in e["postags"]]
for e in dev_data:
    e["tk_words"] = [w2idx.get(w, 0) for w in e["words"]]
    if cfg["use_postags"]:
        e["tk_postags"] = [pos2idx.get(t, 0) for t in e["postags"]]
for e in test_data:
    e["tk_words"] = [w2idx.get(w, 0) for w in e["words"]]
    if cfg["use_postags"]:
        e["tk_postags"] = [pos2idx.get(t, 0) for t in e["postags"]]

num_train_tokens = sum([len(e["words"]) for e in train_data])
num_dev_tokens = sum([len(e["words"]) for e in dev_data])
num_test_tokens = sum([len(e["words"]) for e in test_data])

m = dy.ParameterCollection()
if cfg["model_type"] == "vaswani":
    w2e = m.add_lookup_parameters((num_words, cfg["w_emb_dim"]))
    t2e = m.add_lookup_parameters((num_tags, cfg["t_emb_dim"]))
    input_dim = cfg["w_emb_dim"]
    if cfg["use_postags"]:
        pos2e = m.add_lookup_parameters((num_pos, cfg["pos_emb_dim"]))
        input_dim += cfg["pos_emb_dim"]

    fwd = dy.VanillaLSTMBuilder(1, input_dim, cfg["bilstm_h_dim"], m)
    bwd = dy.VanillaLSTMBuilder(1, input_dim, cfg["bilstm_h_dim"], m)
    lm = dy.VanillaLSTMBuilder(1, cfg["t_emb_dim"], cfg["lm_h_dim"], m)
    c1_Wf = m.add_parameters((cfg["bilstm_h_dim"], cfg["bilstm_h_dim"]))
    c1_Wb = m.add_parameters((cfg["bilstm_h_dim"], cfg["bilstm_h_dim"]))
    # NOTE: dim compute to allow to put different amounts of capacity in the lm and bilstm.
    out_c2_dim = max([cfg["lm_h_dim"], cfg["bilstm_h_dim"]])
    c2_Wlm = m.add_parameters((out_c2_dim, cfg["lm_h_dim"]))
    c2_Wc = m.add_parameters((out_c2_dim, cfg["bilstm_h_dim"]))
    o_W = m.add_parameters((num_tags, out_c2_dim))
    o_b = m.add_parameters(num_tags)
    # assert not (cfg["use_beam_mlp"] and cfg["use_beam_bilstm"])
    # if cfg["use_beam_bilstm"]:
    #     b_fwd = dy.VanillaLSTMBuilder(1, out_c2_dim, out_c2_dim / 2, m)
    #     b_bwd = dy.VanillaLSTMBuilder(1, out_c2_dim, out_c2_dim / 2, m)
    # if cfg["use_beam_mlp"]:
    #     b_W1 = m.add_parameters((out_c2_dim, out_c2_dim))
    #     b_b1 = m.add_parameters(out_c2_dim)
    #     b_W2 = m.add_parameters((out_c2_dim, out_c2_dim))
    #     b_b2 = m.add_parameters(out_c2_dim)
# weaker model that just uses the current word.
elif cfg["model_type"] == 'lm':
    w2e = m.add_lookup_parameters((num_words, cfg["w_emb_dim"]))
    t2e = m.add_lookup_parameters((num_tags, cfg["t_emb_dim"]))
    input_dim = cfg["w_emb_dim"]
    if cfg["use_postags"]:
        pos2e = m.add_lookup_parameters((num_pos, cfg["pos_emb_dim"]))
        input_dim += cfg["pos_emb_dim"]
    lm = dy.VanillaLSTMBuilder(1, cfg["t_emb_dim"], cfg["lm_h_dim"], m)
    W = m.add_parameters((num_tags, input_dim + cfg["lm_h_dim"]))
    b = m.add_parameters(num_tags)

m.set_weight_decay(cfg["weight_decay"])

# #### load pretrained embeddings if necessary
# def load_glove(filepath):
#     lines = tb_io.read_textfile(filepath)
#     num_embs = len(lines)

#     d = len(lines[0].split(' ')) - 1
#     words = []
#     embs = np.zeros((num_embs, d))
#     for i, x in enumerate(lines):
#         y = x.split(' ')
#         w = y[0]
#         e = np.array(y[1:], dtype='float32')

#         embs[i, :] = e
#         words.append(w)

#     return words, embs

# def load_embeddings(embs_type, data_folder, embedding_dim):
#     if embs_type == 'glove':
#         filepath = tb_fs.join_paths([
#             data_folder, 'glove', 'glove.6B',
#             'glove.6B.%dd.txt' % embedding_dim
#         ])
#         words, embs = load_glove(filepath)

#     else:
#         raise ValueError

#     return (words, embs)

# if cfg["use_pretrained_embeddings"]:
#     words, embs = load_embeddings('glove', 'data', cfg["w_emb_dim"])
#     out_embs = w2e.npvalue().T
#     for from_idx, w in enumerate(words):
#         if w in w2idx:
#             to_idx = w2idx[w]
#             # 0.003 serves to equalize the norm of the glove embeddings
#             out_embs[to_idx] = 0.003 * embs[from_idx]
#     w2e.init_from_array(out_embs)

########### MODEL ###########

#### vaswani


def _vaswani_model_init(e):
    w_embs = [w2e[idx] for idx in e["tk_words"]]
    if cfg["use_postags"]:
        pos_embs = [pos2e[idx] for idx in e["tk_postags"]]
        i_embs = [
            dy.concatenate([w_embs[i], pos_embs[i]])
            for i in xrange(len(e["tk_words"]))
        ]
    else:
        i_embs = w_embs

    f_init = fwd.initial_state()
    b_init = bwd.initial_state()
    lm_init = lm.initial_state()

    f_hs = dy.concatenate_cols(f_init.transduce(i_embs))
    b_hs = dy.concatenate_cols(b_init.transduce(reversed(i_embs))[::-1])
    out_c1 = dy.rectify(c1_Wf * f_hs + c1_Wb * b_hs)
    aux_c2 = c2_Wc * out_c1

    m = {
        "aux_c2": aux_c2,
        "beam_lm_states": [lm_init],
        "beam_lm_hs": dy.zeros((cfg["lm_h_dim"], 1)),
        "idx": 0
    }
    if cfg["accumulate_scores"]:
        m["acc_scores"] = dy.zeros((1, 1))

    return m


def _vaswani_model_scores(m):
    out_c2 = dy.rectify(
        dy.colwise_add(c2_Wlm * m["beam_lm_hs"],
                       dy.pick(m["aux_c2"], m["idx"], 1)))

    # if cfg["use_beam_bilstm"]:
    #     _, beam_size_prev = out_c2.dim()[0]
    #     beam_hs = [dy.pick(out_c2, i, 1) for i in xrange(beam_size_prev)]
    #     bf_init = b_fwd.initial_state()
    #     bb_init = b_bwd.initial_state()
    #     bf_hs = dy.concatenate_cols(bf_init.transduce(beam_hs))
    #     bb_hs = dy.concatenate_cols(bb_init.transduce(reversed(beam_hs))[::-1])
    #     out_c2 = dy.concatenate([bf_hs, bb_hs])

    # if cfg["use_beam_mlp"]:
    #     out_b = dy.max_dim(b_W1 * out_c2 + b_b1, 1)
    #     out_c2 = dy.colwise_add(out_c2, dy.rectify(b_W2 * out_b + b_b2))

    scores = o_W * out_c2 + o_b
    scores = dy.transpose(scores)
    if cfg["accumulate_scores"]:
        scores = m["acc_scores"] + scores
        m["scores"] = scores

    return scores


def _vaswani_model_step(m, beam_indices, tag_indices):
    m["beam_lm_states"] = [
        m["beam_lm_states"][b_idx].add_input(t2e[t_idx])
        for (b_idx, t_idx) in izip(beam_indices, tag_indices)
    ]
    m["beam_lm_hs"] = dy.concatenate_cols(
        [x.output() for x in m["beam_lm_states"]])
    m["idx"] = m["idx"] + 1

    if cfg["accumulate_scores"]:
        beam_size_prev, num_tags = m["scores"].dim()[0]
        scores_flat = dy.reshape(m["scores"], (beam_size_prev * num_tags, 1))
        m["acc_scores"] = dy.select_rows(
            scores_flat, beam_indices + tag_indices * beam_size_prev)


##### lm


def _lm_model_init(e):
    w_embs = [w2e[idx] for idx in e["tk_words"]]
    if cfg["use_postags"]:
        pos_embs = [pos2e[idx] for idx in e["tk_postags"]]
        i_embs = [
            dy.concatenate([w_embs[i], pos_embs[i]])
            for i in xrange(len(e["tk_words"]))
        ]
    else:
        i_embs = w_embs

    lm_init = lm.initial_state()

    m = {
        "i_embs": i_embs,
        "beam_lm_states": [lm_init],
        "beam_lm_hs": dy.zeros((cfg["lm_h_dim"], 1)),
        "idx": 0
    }
    if cfg["accumulate_scores"]:
        m["acc_scores"] = dy.zeros((1, 1))
    return m


def _lm_model_scores(m):
    # assert not (cfg["use_beam_bilstm"] or cfg["use_beam_mlp"])

    idx = m["idx"]
    cur_beam_size = len(m["beam_lm_states"])
    q = dy.reshape(m["i_embs"][idx], (input_dim, 1)) * dy.ones(
        (1, cur_beam_size))
    x = dy.concatenate([m["beam_lm_hs"], q])
    scores = W * x + b
    scores = dy.transpose(scores)

    if cfg["accumulate_scores"]:
        scores = m["acc_scores"] + scores
        m["scores"] = scores

    return scores


def _lm_model_step(m, beam_indices, tag_indices):
    m["beam_lm_states"] = [
        m["beam_lm_states"][b_idx].add_input(t2e[t_idx])
        for (b_idx, t_idx) in izip(beam_indices, tag_indices)
    ]
    m["beam_lm_hs"] = dy.concatenate_cols(
        [x.output() for x in m["beam_lm_states"]])
    m["idx"] = m["idx"] + 1

    if cfg["accumulate_scores"]:
        beam_size_prev, num_tags = m["scores"].dim()[0]
        scores_flat = dy.reshape(m["scores"], (beam_size_prev * num_tags, 1))
        m["acc_scores"] = dy.select_rows(
            scores_flat, beam_indices + tag_indices * beam_size_prev)


#### PICKING THE RIGHT FUNCTION
if cfg["model_type"] == 'vaswani':
    model_init = _vaswani_model_init
    model_scores = _vaswani_model_scores
    model_step = _vaswani_model_step

elif cfg["model_type"] == "lm":
    model_init = _lm_model_init
    model_scores = _lm_model_scores
    model_step = _lm_model_step
else:
    raise ValueError

###


def backpointers2preds(backpointers):
    back_idx = 0
    pred_tags = []
    for beam_indices, tag_indices in backpointers[::-1]:
        pred_tags.append(idx2t[tag_indices[back_idx]])
        back_idx = beam_indices[back_idx]
    return pred_tags[::-1]


def vanilla_beam_accuracy(data, beam_size):
    num_correct = 0
    total = 0
    for e in data:
        pred_tags = vanilla_beam_predict(e, beam_size)
        total += len(pred_tags)
        num_correct += sum([g == p for (g, p) in izip(e[tags_key], pred_tags)])
    return float(num_correct) / total


def beam_accuracy(data, beam_size):
    num_correct = 0
    total = 0
    for e in data:
        pred_tags = beam_predict(e, beam_size)
        total += len(pred_tags)
        num_correct += sum([g == p for (g, p) in izip(e[tags_key], pred_tags)])
    return float(num_correct) / total


def logsumexp(x):
    m = np.max(x, axis=-1)
    z = np.log(np.sum(np.exp(x - m[:, None]), axis=-1))
    return m + z


def beam_argtopk(scores, k):
    beam_size, num_tags = scores.shape
    flat_scores = scores.ravel()
    flat_indices = np.argsort(flat_scores)[::-1]
    beam_indices, tag_indices = np.divmod(flat_indices[:k], num_tags)
    return (beam_indices, tag_indices)


def vanilla_beam_predict(e, beam_size, return_backpointers=False):
    dy.renew_cg()

    m = model_init(e)
    beam_acc_log_probs = np.array([0.0])
    backpointers = []
    for i in xrange(len(e["tk_words"])):
        scores = model_scores(m)
        scores_np = scores.npvalue()
        log_probs = scores_np - logsumexp(scores_np)[:, None]
        acc_log_probs = beam_acc_log_probs[:, None] + log_probs

        beam_indices, tag_indices = beam_argtopk(acc_log_probs, beam_size)
        beam_acc_log_probs = acc_log_probs[beam_indices, tag_indices]
        backpointers.append((beam_indices, tag_indices))

        model_step(m, beam_indices, tag_indices)

    pred_tags = backpointers2preds(backpointers)
    if not return_backpointers:
        return pred_tags
    else:
        return pred_tags, backpointers


def beam_predict(e, beam_size, return_backpointers=False):
    dy.renew_cg()

    m = model_init(e)
    backpointers = []
    for i in xrange(len(e["tk_words"])):
        scores = model_scores(m)
        scores_np = scores.npvalue()
        beam_indices, tag_indices = beam_argtopk(scores_np, beam_size)
        backpointers.append((beam_indices, tag_indices))
        model_step(m, beam_indices, tag_indices)

    pred_tags = backpointers2preds(backpointers)
    if not return_backpointers:
        return pred_tags
    else:
        return pred_tags, backpointers


########### LOSSES ###########
# NOTE: for losses, there are two cases, when there is a cost increase and
# when there is not one.
# NOTE: current implementation does not take into account that the number of
# neighbors might be smaller than the size of the beam. this needs to be fixed for
# smaller tag sets. note that this affects the loss
# (if there is nothing outside the beam, the loss becomes zero in some cases,
# e.g., loss_margin_last, upper_bound, perceptron_last, margin_last, ...)
# TODO: address the problem above for smaller tag sets.


# idx is the position for which we are currently predicting for.
def dynet_compute_costs_flat(gold_tags, idx, beam_costs_prev):
    beam_size_prev = beam_costs_prev.shape[0]
    t_idx = gold_tags[idx]
    # computation of the costs for the extensions
    costs = beam_costs_prev.reshape((beam_size_prev, 1)) * np.ones(
        (1, num_tags))
    costs += 1.0
    costs[:, t_idx] -= 1.0
    # NOTE: this is cumbersome because the reshape operation for
    # dynet and numpy is different (column vs row ordering, respectively).
    # accounting for differences in ordering for dynet and numpy.
    return costs.T.ravel()


def dynet_index_flat_to_mat(num_rows, num_cols, idx):
    b_idx = idx % num_rows
    t_idx = int((idx - b_idx) / num_rows)
    return b_idx, t_idx


def dynet_index_mat_to_flat(num_rows, num_cols, b_idx, t_idx):
    return b_idx + t_idx * num_rows


def dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev):
    beam_size_prev = beam_costs_prev.shape[0]
    b_idx = np.argmin(beam_costs_prev)
    t_idx = gold_tags[idx]
    gold_idx = b_idx + t_idx * beam_size_prev
    return gold_idx


### NOTE: make this more consistent in the notation.
def loss_log_neighbors(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    b_idx = np.argmin(beam_costs_prev)
    t_idx = gold_tags[idx]
    gold_idx = b_idx + t_idx * beam_size_prev

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    loss = dy.pickneglogsoftmax(scores_flat, gold_idx)
    return loss


def loss_log_beam(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    gold_idx = dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev)

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    scores_flat_np = scores_flat.npvalue()
    sigma_hat = np.argsort(scores_flat_np)[::-1]
    indices = list(sigma_hat[:beam_size])
    # compute the set I as in the paper
    if gold_idx not in indices:
        indices.append(gold_idx)
    return -dy.pick(dy.log_softmax(scores_flat, restrict=indices), gold_idx)


def loss_cost_sensitive_margin_last(gold_tags, idx, beam_costs_prev, scores,
                                    beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    gold_idx = dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev)

    costs_flat = dynet_compute_costs_flat(gold_tags, idx, beam_costs_prev)

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    scores_flat_np = scores_flat.npvalue()
    sigma_hat = np.argsort(scores_flat_np)[::-1]

    # the beam size for the last transition is one.
    next_beam_size = beam_size if idx < len(gold_tags) - 1 else 1

    # gold_idx is inside the beam, so compare to first outside beam.
    if gold_idx in sigma_hat[:next_beam_size]:
        comp_idx = sigma_hat[next_beam_size]
    # gold_idx is outside the beam, so compare to last in beam.
    else:
        comp_idx = sigma_hat[next_beam_size - 1]

    # NOTE: this can be zero if comp_idx has the same cost as gold_idx (desirable?)
    cost_delta = costs_flat[comp_idx] - costs_flat[gold_idx]
    return cost_delta * dy.rectify(scores_flat[comp_idx] -
                                   scores_flat[gold_idx] + 1.0)


def loss_margin_last(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    gold_idx = dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev)

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    scores_flat_np = scores_flat.npvalue()
    sigma_hat = np.argsort(scores_flat_np)[::-1]

    # the beam size for the last transition is one.
    next_beam_size = beam_size if idx < len(gold_tags) - 1 else 1

    # gold_idx is inside the beam, so compare to first outside beam.
    if gold_idx in sigma_hat[:next_beam_size]:
        comp_idx = sigma_hat[next_beam_size]
    # gold_idx is outside the beam, so compare to last in beam.
    else:
        comp_idx = sigma_hat[next_beam_size - 1]

    return dy.rectify(scores_flat[comp_idx] - scores_flat[gold_idx] + 1.0)


def loss_perceptron_first(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))

    # computation of the index of the best
    gold_idx = dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev)

    # computation of the index that we compare to (first in the beam)
    scores_flat_np = scores_flat.npvalue()
    pred_idx = np.argmax(scores_flat_np)

    return dy.rectify(scores_flat[pred_idx] - scores_flat[gold_idx])


def loss_perceptron_last(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    next_beam_size = beam_size if idx < len(gold_tags) - 1 else 1

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    scores_flat_np = scores_flat.npvalue()
    sigma_hat = np.argsort(scores_flat_np)[::-1]

    # computation of the index of the best
    gold_idx = dynet_get_best_flat_idx(gold_tags, idx, beam_costs_prev)

    # computation of the index that we compare to (last in the beam)
    comp_idx = sigma_hat[next_beam_size - 1]
    return dy.rectify(scores_flat[comp_idx] - scores_flat[gold_idx])


### TODO: this is a bit more tricky with the hinge loss.
# TODO: this is totally wrong because it requires to index the
# thing. there should exist a better way of accomplishing things.
def loss_upper_bound(gold_tags, idx, beam_costs_prev, scores, beam_size):
    beam_size_prev, num_tags = scores.dim()[0]
    next_beam_size = beam_size if idx < len(gold_tags) - 1 else 1

    scores_flat = dy.reshape(scores, (beam_size_prev * num_tags,))
    costs_flat = dynet_compute_costs_flat(gold_tags, idx, beam_costs_prev)

    sigma_star = np.argsort(costs_flat)
    gold_idx = sigma_star[0]
    scores_flat_np = scores_flat.npvalue()
    sigma_hat = np.argsort(scores_flat_np)[::-1]

    scores_delta = scores_flat - scores_flat[gold_idx] + 1.0
    costs_delta = costs_flat - costs_flat[gold_idx]
    # mask those that are inside the beam.
    costs_delta[sigma_star[:next_beam_size]] = 0.0
    deltas = dy.cmult(dy.inputTensor(costs_delta), scores_delta)
    return dy.max_dim(deltas)


########### MODEL TRAINING ###########


# TODO: change costs from ints to floats.
def train_beam_graph(e, beam_size, traj_type, loss_fn):
    dy.renew_cg()

    tags = e[tk_tags_key]
    m = model_init(e)
    beam_costs_prev = np.array([0], dtype="int")
    beam_costs = []
    losses = []
    for i in xrange(len(e["tk_words"])):
        scores = model_scores(m)

        # transition
        scores_np = scores.npvalue()
        beam_indices, tag_indices = beam_argtopk(scores_np, beam_size)

        beam_costs_cur = beam_costs_prev[beam_indices] + (tag_indices !=
                                                          tags[i]).astype('int')

        # compute the loss if there is score accumulation or always
        next_beam_size = beam_size if i < len(e["tk_words"]) - 1 else 1
        if (not cfg["update_only_on_cost_increase"]) or (
                cfg["update_only_on_cost_increase"] and
                beam_costs_prev.min() < beam_costs_cur[:next_beam_size].min()):
            loss = loss_fn(tags, i, beam_costs_prev, scores, beam_size)
            losses.append(loss)

        if traj_type == "stop":
            if beam_costs_cur.min() > 0:
                break
        elif traj_type == "continue":
            pass
        elif traj_type == "reset":
            if beam_costs_cur.min() > 0:
                b_gold_idx = beam_costs_prev.argmin()
                beam_indices = np.array([b_gold_idx], dtype='int')
                tag_indices = np.array([tags[i]], dtype='int')
                beam_costs_cur = np.array([0], dtype='int')
        elif traj_type == "reset_multiple":
            # NOTE: this is similar to the reset option. replace the last element
            # in the beam with the correct one.
            if beam_costs_cur.min() > 0:
                b_gold_idx = beam_costs_prev.argmin()
                beam_indices[-1] = b_gold_idx
                tag_indices[-1] = tags[i]
                beam_costs_cur[-1] = beam_costs_prev[b_gold_idx]
                # this should be zero
                # assert beam_costs_prev[-1] == 0
        # NOTE: there is probably a less repetitive way of doing this.
        elif traj_type == "oracle":
            t_idx = tags[i]
            beam_size_prev = beam_costs_prev.shape[0]
            costs = beam_costs_prev.reshape((beam_size_prev, 1)) * np.ones(
                (1, num_tags))
            costs += 1.0
            costs[:, t_idx] -= 1.0
            beam_indices, tag_indices = beam_argtopk(-costs, beam_size)
            beam_costs_cur = beam_costs_prev[beam_indices] + (
                tag_indices != tags[i]).astype('int')

        else:
            raise ValueError

        beam_costs.append(beam_costs_cur)
        beam_costs_prev = beam_costs_cur
        model_step(m, beam_indices, tag_indices)

    if len(losses) > 0:
        return dy.esum(losses)
    else:
        return dy.zeros(1)


#
def train_model_with_config():
    import research_toolbox.tb_logging as tb_lg

    if cfg["optimizer_type"] == "sgd":
        trainer = dy.SimpleSGDTrainer(m, cfg["step_size_start"])
    elif cfg["optimizer_type"] == "adam":
        trainer = dy.AdamTrainer(m, cfg["step_size_start"])
    elif cfg["optimizer_type"] == "sgd_mom":
        trainer = dy.MomentumSGDTrainer(m, cfg["step_size_start"])
    else:
        raise ValueError
    trainer.set_sparse_updates(0)

    # restarting from a checkpoint if it exists.
    # optimizer state is not kept.
    ckpt_filepath = cfg["out_folder"] + "/checkpoint.json"
    if tb_fs.file_exists(ckpt_filepath):
        log_d = tb_io.read_jsonfile(ckpt_filepath)
        current_epoch = len(log_d["dev_acc"])
        best_dev_acc = np.max(log_d["dev_acc"])
        m.populate(cfg["out_folder"] + '/model.ckpt')
    else:
        current_epoch = 0
        best_dev_acc = 0.0

        log_d = {
            'dev_acc': [],
            'avg_loss': [],
            'train_tks/sec': [],
            'eval_tks/sec': [],
            'secs_per_epoch': [],
            "lr": []
        }
        if cfg["debug"] or cfg["compute_train_acc"]:
            log_d["train_acc"] = []

    if cfg["loss_type"] == "log_neighbors":
        loss_fn = loss_log_neighbors
    elif cfg["loss_type"] == "log_beam":
        loss_fn = loss_log_beam
    elif cfg["loss_type"] == "cost_sensitive_margin_last":
        loss_fn = loss_cost_sensitive_margin_last
    elif cfg["loss_type"] == "margin_last":
        loss_fn = loss_margin_last
    elif cfg["loss_type"] == "perceptron_first":
        loss_fn = loss_perceptron_first
    elif cfg["loss_type"] == "perceptron_last":
        loss_fn = loss_perceptron_last
    elif cfg["loss_type"] == "upper_bound":
        loss_fn = loss_upper_bound
    else:
        raise ValueError

    cfg_accuracy = lambda data: beam_accuracy(data, cfg["beam_size"])
    cfg_train_graph = lambda e: train_beam_graph(e, cfg["beam_size"], cfg[
        "traj_type"], loss_fn)

    for epoch in range(current_epoch, cfg["num_epochs"]):
        if cfg["step_size_schedule_type"] == 'fixed':
            lr = cfg["step_size_start"]
        elif cfg["step_size_schedule_type"] == 'cosine':
            lr = cosine_get_lr(cfg["step_size_start"], cfg["step_size_end"],
                               cfg["num_epochs"], epoch)
        else:
            raise ValueError
        log_d['lr'].append(lr)

        trainer.learning_rate = lr

        acc_loss = 0.0
        random.shuffle(train_data)
        epoch_timer = tb_lg.TimeTracker()
        train_timer = tb_lg.TimeTracker()
        for i, e in enumerate(train_data):
            if i % cfg["print_every_num_examples"] == 0 and i > 0:
                print "Epoch %d - Example %d/%d" % (epoch, i, len(train_data))
            loss = cfg_train_graph(e)
            acc_loss += loss.value()
            loss.backward()
            trainer.update()

        log_d["avg_loss"].append(acc_loss / len(train_data))
        log_d["train_tks/sec"].append(num_train_tokens /
                                      train_timer.time_since_start())
        eval_timer = tb_lg.TimeTracker()
        # log_d['train_acc'].append(accuracy(train_data))
        log_d['dev_acc'].append(cfg_accuracy(dev_data))
        # log_d['test_acc'].append(accuracy(test_data))
        log_d['eval_tks/sec'].append((  #len(train_data) +
            num_dev_tokens
            # + num_test_tokens
        ) / eval_timer.time_since_start())
        log_d["secs_per_epoch"].append(epoch_timer.time_since_start())
        if cfg["debug"] or cfg["compute_train_acc"]:
            train_acc = cfg_accuracy(train_data)
            print "train_acc: ", train_acc
            log_d["train_acc"].append(train_acc)
        pprint({k: vs[-1] for k, vs in log_d.iteritems()})

        if best_dev_acc < log_d["dev_acc"][-1]:
            best_dev_acc = log_d["dev_acc"][-1]
            m.save(cfg["out_folder"] + '/best_model.ckpt')
        tb_io.write_jsonfile(log_d, cfg["out_folder"] + "/checkpoint.json")
        m.save(cfg["out_folder"] + '/model.ckpt')

    results_filepath = cfg["out_folder"] + "/results.json"
    if not tb_fs.file_exists(results_filepath):
        m.populate(cfg["out_folder"] + '/best_model.ckpt')
        log_d['test_acc'] = cfg_accuracy(test_data)
        tb_io.write_jsonfile(log_d, cfg["out_folder"] + "/results.json")


if __name__ == "__main__":
    # train_model()
    if '--train' in sys.argv:
        train_model_with_config()
    elif '--compute_vanilla_beam_accuracy' in sys.argv:
        m.populate(cfg['out_folder'] + '/model.ckpt')
        for s in [1, 2, 4, 8]:
            print(s, vanilla_beam_accuracy(dev_data, s),
                  vanilla_beam_accuracy(test_data, s))
    elif '--compute_beam_accuracy' in sys.argv:
        m.populate(cfg['out_folder'] + '/model.ckpt')
        print(beam_accuracy(dev_data, cfg["beam_size"]),
              beam_accuracy(test_data, cfg["beam_size"]))
    else:
        raise ValueError
