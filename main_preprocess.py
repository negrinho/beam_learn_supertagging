# -*- coding: utf-8 -*-

import research_toolbox.tb_io as tb_io
import research_toolbox.tb_filesystem as tb_fs


def read_supertagging_auto_file(filepath):
    lines = tb_io.read_textfile(filepath)
    num_examples = len(lines) / 2
    lst = []
    for i in xrange(num_examples):
        postags = []
        supertags = []
        words = []
        s = lines[2 * i + 1]
        while True:
            start_idx = s.find("<L ")
            if start_idx == -1:
                break
            end_idx = s.find(">)") + 1
            node = s[start_idx + 3:end_idx - 2]
            super_t, _, pos_t, tk, _ = node.split(' ')

            postags.append(pos_t)
            supertags.append(super_t)
            words.append(tk)

            # skips potential whitespace
            s = s[end_idx + 1:]

        example_id, parser_id, num_parses = lines[2 * i].split(' ')
        lst.append({
            "example_id": example_id.split("=")[1],
            "parser_id": parser_id.split("=")[1],
            "num_parses": int(num_parses.split("=")[1]),
            "words": words,
            "postags": postags,
            "supertags": supertags
        })

    return lst


if __name__ == "__main__":
    # Path to CCG Bank AUTO folder.
    folderpath = "data/ccgbank_1_1/data/AUTO/"
    filepath_lst = tb_fs.list_files(folderpath, recursive=True)
    examples = []
    for fpath in filepath_lst:
        examples.extend(read_supertagging_auto_file(fpath))

    train_examples = []
    dev_examples = []
    test_examples = []
    idx = len("wsj_")
    for e in examples:
        section_id = int(e["example_id"][idx:idx + 2])
        if section_id >= 2 and section_id <= 21:
            train_examples.append(e)
        elif section_id == 0:
            dev_examples.append(e)
        elif section_id == 23:
            test_examples.append(e)
        else:
            continue

    print len(train_examples), len(dev_examples), len(test_examples)
    # Paths for the output files
    tb_fs.create_folder("data/supertagging", abort_if_exists=False)
    tb_io.write_jsonlogfile("data/supertagging/train.jsonl", train_examples)
    tb_io.write_jsonlogfile("data/supertagging/dev.jsonl", dev_examples)
    tb_io.write_jsonlogfile("data/supertagging/test.jsonl", test_examples)