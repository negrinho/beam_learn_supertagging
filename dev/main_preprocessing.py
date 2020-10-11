#% to process other tagging data (ptb, conll 2000, conll 2003) into the format used to run the model in the paper.
# while this data has not been used in the paper, it might be of interest to consider.

import research_toolbox.tb_io as tb_io


def load_datafile(filepath, column_indices, column_names):
    with open(filepath, 'r') as f:
        data = []
        e = {k: [] for k in column_names}
        k0 = column_names[0]
        for line in f:
            line = line.strip()
            if line == '' and len(e[k0]) > 0:
                data.append(e)
                e = {k: [] for k in column_names}
            else:
                fields = line.split()
                for k, i in zip(column_names, column_indices):
                    e[k].append(fields[i])
        return data


def process_conll2000(in_filepath, out_filepath):
    column_indices = [0, 2, 1]
    column_names = ["words", "chunk_tags", "postags"]
    data = load_datafile(in_filepath, column_indices, column_names)
    tb_io.write_jsonlogfile(out_filepath, data)


def process_conll2003(in_filepath, out_filepath):
    column_indices = [0, 3, 1]
    column_names = ["words", "ner_tags", "postags"]
    data = load_datafile(in_filepath, column_indices, column_names)
    tb_io.write_jsonlogfile(out_filepath, data)


# NOTE: this cannot be used with the current code.
def process_ptb(in_filepath, out_filepath):
    column_indices = [0, 1]
    column_names = ["words", "postags"]
    data = load_datafile(in_filepath, column_indices, column_names)
    tb_io.write_jsonlogfile(out_filepath, data)


#%%
process_conll2000('data/raw_conll_2000/train.txt', 'data/conll2000/train.jsonl')
process_conll2000('data/raw_conll_2000/test.txt', 'data/conll2000/test.jsonl')
#%%
process_conll2003('data/raw_conll_2003/eng.train.converted',
                  'data/conll2003/train.jsonl')
process_conll2003('data/raw_conll_2003/eng.testa.converted',
                  'data/conll2003/dev.jsonl')
process_conll2003('data/raw_conll_2003/eng.testb.converted',
                  'data/conll2003/test.jsonl')
# %%
process_ptb('data/raw_ptb/train.txt', 'data/ptb/train.jsonl')
process_ptb('data/raw_ptb/dev.txt', 'data/ptb/dev.jsonl')
process_ptb('data/raw_ptb/test.txt', 'data/ptb/test.jsonl')
