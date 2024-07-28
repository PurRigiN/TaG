# implementation of data reader of different datasets
from itertools import accumulate
import json
from tqdm import tqdm
import torch
import numpy as np
import os
import spacy
from spacy.tokens import Doc

from modules.mention_extraction import tag

# find project abs dir
root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(root_dir)

docred_rel2id = json.load(open(os.path.join(root_dir, "data/docred/rel2id.json"), "r"))
docred_id2rel = {v: k for k, v in docred_rel2id.items()}

def read_dataset(tokenizer, split='train_annotated', dataset='docred', task='me'):
    if task == 'me':
        return read_me_data(tokenizer=tokenizer, split=split, dataset=dataset)
    if task == 'gc':
        return read_gc_data(tokenizer=tokenizer, split=split, dataset=dataset)
    raise ValueError("Unknown task type.")

def read_me_data(tokenizer, split='train_annotated', dataset='docred'):
    i_line = 0
    features = []
    bias_mentions, bias_entities = 0, 0    # count the mention that been discard
    cnt_mention = 0
    data_path = 'data/{}/{}.json'.format(dataset, split)
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))
    if dataset == 'docred' or dataset == 're-docred':
        rel2id = docred_rel2id
    else:
        raise ValueError("Unknown dataset.")

    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    # for every doc
    # merge all sentences into one tokenized sentence
    for sample in tqdm(data, desc='Data'):
        i_line += 1
        sents = []
        doc_map = []
        entities = sample['vertexSet']
        spans = []
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}    # map the old index to the new index
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            sent_map[i_t + 1] = len(sents)
            doc_map.append(sent_map)
        label = [tag["O"] for i in range(len(sents))]
        entity_len = []
        for e in entities:
            flag_entity = False
            e_new = set()   # first: remove duplicate mentions for single entity
            for m in e:
                e_new.add((m["sent_id"], m["pos"][0], m["pos"][1]))
            entity_len.append(len(e_new))  # save entity lens to later construct cr_label
            cnt_mention += len(e_new)
            for m in e_new:
                flag_mention = False
                start, end = doc_map[m[0]][m[1]], doc_map[m[0]][m[2]]
                # add this span to mr_spans
                spans.append((start + offset, end + offset))
                
                # check if has been label in range(start, end)
                for j in range(start, end):
                    if label[j] != tag["O"]:
                        bias_mentions += 1
                        flag_mention = True
                        break
                if flag_mention:
                    flag_entity = True
                    continue        # this "continue" stop labeling if this mention is bias mention

                # add BI label
                label[start] = tag["B"]
                for j in range(start + 1, end):
                    label[j] = tag["I"]

            if flag_entity:
                bias_entities += 1

        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        label = [tag["O"]] + label + [tag["O"]]     # [CLS]+tokens+[SEP]
        # collect re_triples to find useful mentions
        accumulate_entity_len = [0] + [i for i in accumulate(entity_len)]
        re_triples = []
        if split != "test":
            for hrt in sample["labels"]:
                h = hrt["h"]
                t = hrt["t"]
                r = rel2id[hrt["r"]]
                re_triples.append({"h": h, "r": r, "t": t})
        # collect useful mention: mentions of entities in relation triples
        useful_entity = set()
        useful_mention = []
        for hrt in re_triples:
            useful_entity.add(hrt["h"])
            useful_entity.add(hrt["t"])
        for e in useful_entity:
            for i in range(accumulate_entity_len[e], accumulate_entity_len[e + 1]):
                useful_mention.append(spans[i])
        feature = {"input_ids": input_ids, "label": label, "spans": spans, 
                    "doc_map": doc_map, "doc_len": len(input_ids), "useful_mention": useful_mention}
        features.append(feature)

    print("# of documents:\t\t{}".format(i_line))
    print("# of mention bias:\t{}".format(bias_mentions))
    print("# of entity bias:\t{}".format(bias_entities))
    print("# of mentions:\t\t{}".format(cnt_mention))
    return features

def read_gc_data(tokenizer, split='train_annotated', dataset='docred'):
    i_line = 0
    features = []
    data_path = f'data/{dataset}/{split}_gc.json'
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))
    # init spacy
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe('coreferee')

    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    for sample in tqdm(data, desc='Data'):
        # ----process spacy----
        sents_len_list = [len(sent) for sent in sample['sents']]
        word_list = []
        for sent in sample['sents']:
            word_list.extend(sent)
        doc_obj = Doc(nlp.vocab, words=word_list)
        doc = nlp(doc_obj)
        # ----process spacy----
        i_line += 1
        sents = []
        doc_map = []
        span_start, span_end = [], []
        raw_spans = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in sample['spans']]
        for s in raw_spans:
            span_start.append(s[0])
            span_end.append(s[1])
        # get doc_map, input_ids
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}    # map the old index to the new index
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                # add "*" token before and after mention
                if (i_s, i_t) in span_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t + 1) in span_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            sent_map[i_t + 1] = len(sents)
            doc_map.append(sent_map)
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        
        raw_anaphors = [] # 每个元素是一个二元元组，表明在anaphor在哪句里面的哪个位置, index从0开始
        link_span_anaphor = [] # 每个元素是一个二元元组，表示第几个span对应第几个anaphor, index从0开始
        for chain in doc._.coref_chains:
            chain_index_i = 0
            in_loop_index_i = 0
            while chain_index_i != len(chain):
                if len(chain[chain_index_i]) > 1 and in_loop_index_i < len(chain[chain_index_i]):
                    index = chain[chain_index_i][in_loop_index_i]
                    in_loop_index_i += 1
                else:
                    in_loop_index_i = 0
                    index = chain[chain_index_i][0]
                    chain_index_i += 1
                token_obj = doc[index]
                # 如果token是专有名词
                if token_obj.pos_ == 'PROPN':
                    # 查找是否出现在span中
                    sent_id, index_in_sent = find_sent_id_location(index, sents_len_list)
                    for i, singal_raw_span in enumerate(raw_spans):
                        if singal_raw_span[0][0] == sent_id and singal_raw_span[0][1] <= index_in_sent < singal_raw_span[1][1]:
                            # 该专有名词出现在span中
                            # 该chain中的所有指代(普通名词/指代词)，没有在anaphors中的，加入到anaphors中
                            # 并记录哪个span与哪个anaphor对应
                            chain_index_j = 0
                            in_loop_index_j = 0
                            while chain_index_j != len(chain):
                                if len(chain[chain_index_j]) > 1 and in_loop_index_j < len(chain[chain_index_j]):
                                    idx = chain[chain_index_j][in_loop_index_j]
                                    in_loop_index_j += 1
                                else:
                                    in_loop_index_j = 0
                                    idx = chain[chain_index_j][0]
                                    chain_index_j += 1
                                tk_obj = doc[idx]
                                if tk_obj.pos_ == 'NOUN' or  tk_obj.pos_ == 'PRON':
                                    link_score = 0.5
                                    # 找分数
                                    antecedent_index_in_doc = token_obj.i
                                    for potential_refered in tk_obj._.coref_chains.temp_potential_refereds:
                                        if potential_refered.root_index == antecedent_index_in_doc:
                                            link_score = potential_refered.temp_score
                                            break
                                    singal_ana = find_sent_id_location(idx, sents_len_list)
                                    # 查找是否已经加入到raw_anaphors中
                                    index_anaphors = -1
                                    for temp_index, temp_anaphor in enumerate(raw_anaphors):
                                        if temp_anaphor == singal_ana :
                                            index_anaphors = temp_index
                                            break
                                    if index_anaphors == -1:
                                        raw_anaphors.append(singal_ana)
                                        link_span_anaphor.append((i, len(raw_anaphors) - 1, link_score))
                                    else:
                                        link_span_anaphor.append((i, index_anaphors, link_score))

        # get spans
        # spans 的格式：
        # 是一个列表，每一个元素都是一个2元元组tuple，格式为(start, end)，表示一个提及的span起始到末尾
        # 显然，长度就等于这句话里面的提及数量
        spans = []
        for rs in raw_spans:
            start = doc_map[rs[0][0]][rs[0][1]]
            end = doc_map[rs[1][0]][rs[1][1]]
            spans.append((start + offset, end + offset))
        # get anaphors
        # 每一个元素都代表anaphors在分词之后的句子中的位置
        anaphors = []
        for anaphor in raw_anaphors:
            start = doc_map[anaphor[0]][anaphor[1]]
            anaphors.append(start + offset)

        num_spans = len(spans)
        num_anaphors = len(anaphors)

        # get syntax graph
        syntax_graph = torch.zeros(num_spans + num_anaphors, num_spans + num_anaphors)
        for i in range(num_spans):
            for j in range(num_spans):
                if raw_spans[i][0][0] == raw_spans[j][0][0]:
                    syntax_graph[i][j] = 1
        for i in range(num_spans):
            for j in range(num_spans, num_spans + num_anaphors):
                if raw_spans[i][0][0] == raw_anaphors[j-num_spans][0]:
                    syntax_graph[i][j] = 1
                    syntax_graph[j][i] = 1
        for i in range(num_spans, num_spans + num_anaphors):
            for j in range(num_spans, num_spans + num_anaphors):
                if raw_anaphors[i-num_spans][0] == raw_anaphors[j-num_spans][0]:
                    syntax_graph[i][j] = 1
        # self loop
        for i in range(num_spans + num_anaphors):
            syntax_graph[i][i] = 1
        
        # get clusters and relations
        # if mention not in predictions, add -1 as index
        clusters, relations = [], []
        cr_label, re_label, re_table_label = None, None, None
        hts = None
        hts_table = None
        cr_table_label = None
        if 'labels' in sample:
            for hrt in sample['labels']:
                h = hrt['h']
                t = hrt['t']
                r = docred_rel2id[hrt['r']]
                relations.append({'h': h, 'r': r, 't': t})
        entities = sample['vertexSet']
        for e in entities:
            cluster = []
            for m in e:
                ms = ((m['sent_id'], m['pos'][0]), (m['sent_id'], m['pos'][1]))
                # build a cluster indexer (from ground truth to ME result) 
                # for those is recognized in ME
                if ms not in raw_spans:
                    cluster.append(-1)
                else:
                    cluster.append(raw_spans.index(ms))
            # cluster = list(set(cluster))
            clusters.append(cluster)
        # construct label, by utilizing mapping
        if 'train' in split:
            # construct ht_to_r dict
            ht_to_r = dict()
            for hrt in relations:
                h, r, t = hrt['h'], hrt['r'], hrt['t']
                if (h, t) in ht_to_r:
                    ht_to_r[(h, t)].append(r)
                else:
                    ht_to_r[(h, t)] = [r]
            # construct label
            entity_len = sample['entity_len']
            accumulate_entity_len = [0] + [i for i in accumulate(entity_len)]

            table_cr_table_label = [[0 for i in range(num_spans+num_anaphors)] for j in range(num_spans+num_anaphors)]  # element: 0 or 1
            table_re_table_label = [[0 for i in range(num_spans+num_anaphors)] for j in range(num_spans+num_anaphors)]  # element: 0 or 1
            table_cr_label = [[0 for i in range(num_spans)] for j in range(num_spans)]  # element: 0 or 1
            table_re_label = [[[1]+[0]*(len(docred_rel2id)-1) for i in range(num_spans)] for j in range(num_spans)] # element: list of relation label
            # cr
            for index_e in range(len(entities)):
                for m1 in range(accumulate_entity_len[index_e], accumulate_entity_len[index_e+1]):
                    for m2 in range(accumulate_entity_len[index_e], accumulate_entity_len[index_e+1]):
                        # 给每一个实体内的mention之间的关系添加双向边，表示共指
                        # 包括一阶段和二阶段
                        table_cr_table_label[m1][m2] = 1
                        table_cr_label[m1][m2] = 1
            for (link_m, link_a, link_score) in link_span_anaphor:
                # 对于每一个anaphor，找到对应的mention，然后找到所属的entity
                link_e = find_entity_by_mention(link_m, entity_len)
                # 将该anaphor与entity内的所有mention之间的关系添加label
                for m1 in range(accumulate_entity_len[link_e], accumulate_entity_len[link_e+1]):
                    table_cr_table_label[m1][link_a+num_spans] = link_score
                    table_cr_table_label[link_a+num_spans][m1] = link_score
            for i in range(num_spans, num_spans + num_anaphors):
                for j in range(num_spans, num_spans + num_anaphors):
                    # 对于每一对anaphor
                    for m in range(num_spans):
                        # 如果存在一个mention，与两个anaphor都有关系，那么两个anaphor之间也有关系
                        if table_cr_table_label[i][m] > 0 and table_cr_table_label[j][m] > 0:
                            temp_score = table_cr_table_label[i][m] * table_cr_table_label[j][m]
                            table_cr_table_label[i][j] = temp_score
                            table_cr_table_label[j][i] = temp_score
                            break
            # re
            for h_e in range(len(entities)):
                for t_e in range(len(entities)):
                    if (h_e, t_e) in ht_to_r:
                        relation = [0] * len(docred_rel2id)
                        for r in ht_to_r[(h_e, t_e)]:
                            relation[r] = 1
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                table_re_label[h_m][t_m] = relation
                                table_re_table_label[h_m][t_m] = 1
                                table_re_table_label[t_m][h_m] = 1
                        for a in range(num_spans, num_spans+num_anaphors):
                            if table_cr_table_label[accumulate_entity_len[h_e]][a] > 0:
                                temp_score = table_cr_table_label[accumulate_entity_len[h_e]][a]
                                # 所有与h_e有指代关系的anaphor，将它们与t_e的mention相连接，表示存在关系，双向
                                for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                    table_re_table_label[a][t_m] = temp_score
                                    table_re_table_label[t_m][a] = temp_score
                        for a in range(num_spans, num_spans+num_anaphors):
                            if table_cr_table_label[accumulate_entity_len[t_e]][a] > 0:
                                temp_score = table_cr_table_label[accumulate_entity_len[t_e]][a]
                                # 所有与t_e有指代关系的anaphor，将它们与h_e的mention相连接，表示存在关系，双向
                                for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                                    table_re_table_label[a][h_m] = temp_score
                                    table_re_table_label[h_m][a] = temp_score
                        for a in range(num_spans, num_spans+num_anaphors):
                            if table_cr_table_label[accumulate_entity_len[h_e]][a] > 0:
                                for b in range(num_spans, num_spans+num_anaphors):
                                    if table_cr_table_label[accumulate_entity_len[t_e]][b] > 0:
                                        temp_score = table_cr_table_label[accumulate_entity_len[h_e]][a] * table_cr_table_label[accumulate_entity_len[t_e]][b]
                                        table_re_table_label[a][b] = temp_score
                                        table_re_table_label[b][a] = temp_score
                    # else: 
                        # 没有关系的两个实体，不需要添加关系
                        # 在初始化的时候，就是[1]+[0]*(len(docred_rel2id)-1)，不需要操作

            # anaphors_scores = [1.0   for i in range(num_anaphors)]
            # anaphors_if_cal = [False for i in range(num_anaphors)]
            # for (link_m, link_a, link_score) in link_span_anaphor:
            #     # 如果计算过了就下一个link
            #     if anaphors_if_cal[link_a]:
            #         continue
            #     else:
            #         anaphors_if_cal[link_a] = True
            #     cur_anaphor = raw_anaphors[link_a]
            #     # 依据对应的mention，找到所属的entity
            #     link_e = find_entity_by_mention(link_m, entity_len)
            #     # 对于所有mention
            #     contain_entity_mention = False
            #     contain_other_mention = False
            #     for m in range(num_spans):
            #         if m in range(accumulate_entity_len[link_e], accumulate_entity_len[link_e+1]):
            #             if cur_anaphor[0] == raw_spans[m][0][0]:
            #                 contain_entity_mention = True
            #         else:
            #             if cur_anaphor[0] == raw_spans[m][0][0]:
            #                 contain_other_mention = True
            #     if not contain_entity_mention and contain_other_mention:
            #         # 同句内不存在该实体本身的提及，又存在其他实体的提及，重要性最高
            #         score = 1.0
            #     elif contain_entity_mention and contain_other_mention:
            #         # 同句内既存在该实体本身的提及，又存在其他实体的提及，重要性次之
            #         score = 0.6
            #     else:
            #         score = 0.2
            #     anaphors_scores[link_a] = score

            # process_vector = [1.0 for i in range(num_spans)] + anaphors_scores
            # process_vector = torch.tensor(process_vector)

            # # syntax_graph
            # syntax_graph = syntax_graph * process_vector.unsqueeze(0) * process_vector.unsqueeze(1)
            
            # # table_cr_table_label
            # table_cr_table_label = torch.tensor(table_cr_table_label) * process_vector.unsqueeze(0) * process_vector.unsqueeze(1)
            # table_cr_table_label = table_cr_table_label.cpu().tolist()
            # # table_re_table_label
            # table_re_table_label = torch.tensor(table_re_table_label) * process_vector.unsqueeze(0) * process_vector.unsqueeze(1)
            # table_re_table_label = table_re_table_label.cpu().tolist()

            hts_table, cr_table_label, re_table_label = [], [], []
            hts, cr_label, re_label = [], [], []
            for i in range(num_spans+num_anaphors):
                for j in range(num_spans+num_anaphors):
                    hts_table.append([i, j])
                    cr_table_label.append(table_cr_table_label[i][j])
                    re_table_label.append(table_re_table_label[i][j])
            for i in range(num_spans):
                for j in range(num_spans):
                    hts.append([i, j])
                    cr_label.append(table_cr_label[i][j])
                    re_label.append(table_re_label[i][j])
        syntax_graph[syntax_graph == 0] = -1e30
        syntax_graph = torch.softmax(syntax_graph, dim=-1)
        syntax_graph = syntax_graph.cpu().tolist()

        feature = {"input_ids": input_ids, "spans": spans, "hts": hts,
                   "cr_label": cr_label, "cr_clusters": clusters,
                   "re_label": re_label, "re_triples": relations, "re_table_label": re_table_label,
                   "hts_table": hts_table, 
                   "cr_table_label": cr_table_label,
                   "anaphors": anaphors,
                   "syntax_graph": syntax_graph,    # contain self loop
                   "vertexSet": entities,
                   "title": sample["title"],
                   'raw_spans': raw_spans,
                   'sents': sample['sents'],
                   'labels': sample['labels'] if 'labels' in sample else None,}
        features.append(feature)

    print("# of documents:\t\t{}.".format(i_line))
    return features

def find_sent_id_location(index, sents_len_list):
    """
    a fun to find  which sent and index the anaphor is located in
    """
    index_in_sent = index
    for i, sent_len in enumerate(sents_len_list):
        if index_in_sent < sent_len:
            return (i, index_in_sent)
        index_in_sent -= sent_len
    raise ValueError('Could not find the index of anaphor in the sents')

def find_entity_by_mention(mention, entity_len_list):
    """
    a fun to find which entity the mention belongs to
    """
    index = mention
    for i, entity_len in enumerate(entity_len_list):
        if index < entity_len:
            return i
        index -= entity_len
    raise ValueError('Could not find the entity of mention')