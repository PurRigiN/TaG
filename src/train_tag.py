import argparse
import wandb
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from apex import amp

from model import Table2Graph
from evaluation import *
from data import read_dataset, docred_rel2id, docred_id2rel

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="docred", type=str)

    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--num_gcn_layers", default=3, type=int)
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="alpha parameter for cr/re loss ratio.")
    parser.add_argument("--beta", default=1.0, type=float,
                        help="beta parameter for ATLoss.")
    parser.add_argument("--rho", default=0.1, type=float,
                        help="rho parameter for levenstein decoding.")

    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_epoch", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--evaluation_start_epoch", default=-1, type=int,
                        help="Epoch that begin to evaluate.")
    parser.add_argument("--report_save_folder", default="", type=str,
                        help="Path the analyze report to be saved.")
    parser.add_argument("--curriculum_pre25", type=float,
                        help="Curriculum learning threshold of pre25")
    parser.add_argument("--curriculum_pre50", type=float,
                        help="Curriculum learning threshold of pre50")
    parser.add_argument("--curriculum_pre75", type=float,
                        help="Curriculum learning threshold of pre75")
    parser.add_argument("--trust_threshold", default=0.0, type=float,
                        help="consider anaphor links that higher than trust_threshold")
    parser.add_argument("--num_axial_layers", default=2, type=int)
    parser.add_argument("--no_anaphor", default=False, type=bool)

    parser.add_argument("--device", default="cuda:0", type=str,
                        help="The running device.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--notes", type=str, default="")

    return parser.parse_args()

def train(args, model: Table2Graph, train_features, dev_features, test_features=None):
    def finetune(train_features, optimizer, num_epoch, num_steps):
        best_score = -1
        features = train_features[0]
        dataloader = DataLoader(features, batch_size=args.train_batch_size, 
                                shuffle=True, collate_fn=collate_fn)
        train_iterator = range(num_epoch)
        total_steps = (len(dataloader) // args.gradient_accumulation_steps) * num_epoch
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        curriculum_epoch = int(num_epoch / 4)
        for epoch in train_iterator:
            if args.curriculum_pre25 != None and args.curriculum_pre50 != None and args.curriculum_pre75 != None:
                if epoch == (curriculum_epoch * 1):
                    features = train_features[1]
                    dataloader = DataLoader(features, batch_size=args.train_batch_size, 
                                            shuffle=True, collate_fn=collate_fn)
                elif epoch == (curriculum_epoch * 2):
                    features = train_features[2]
                    dataloader = DataLoader(features, batch_size=args.train_batch_size, 
                                            shuffle=True, collate_fn=collate_fn)
                elif epoch == (curriculum_epoch * 3):
                    features = train_features[3]
                    dataloader = DataLoader(features, batch_size=args.train_batch_size, 
                                            shuffle=True, collate_fn=collate_fn)
            model.zero_grad()
            cum_loss = torch.tensor(0.0).to(args.device)
            for step, batch in tqdm(enumerate(dataloader), desc='train epoch {}'.format(epoch)):
                model.train()
                inputs = {"input_ids": batch["input_ids"].to(args.device),
                          "attention_mask": batch["attention_mask"].to(args.device),
                          "cr_label": batch["cr_label"].to(args.device),
                          "re_label": batch["re_label"].to(args.device),
                          "cr_table_label": batch["cr_table_label"].to(args.device),
                          "re_table_label": batch["re_table_label"].to(args.device),
                          "spans": batch["spans"],
                          "hts_table": batch["hts_table"],
                          "anaphors": batch["anaphors"],
                          "syntax_graph": batch["syntax_graph"].to(args.device),}
                loss = model.compute_loss(**inputs)
                loss = loss / args.gradient_accumulation_steps
                cum_loss += loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    wandb.log({"loss": cum_loss.item()}, step=num_steps)
                    cum_loss = torch.tensor(0.0).to(args.device)
                if step == len(dataloader) - 1 and epoch >= args.evaluation_start_epoch: 
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    if test_features is not None:
                        test_score, test_output = evaluate(args, model, test_features, tag="re-docred-test")
                        dev_output.update(test_output)
                    # train_score, train_output = evaluate(args, model, train_features, tag="train")
                    # dev_output.update(train_output)
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != "":
                            save_dir = os.path.dirname(args.save_path)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)
                            torch.save(model.state_dict(), args.save_path)                    
        return num_steps
    
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "bert" in n], "lr": 3e-5},
        {"params": [p for n, p in model.named_parameters() if not "bert" in n], },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_epoch, num_steps)

def evaluate(args, model: Table2Graph, features, tag=""):
    model.eval()
    # load vertexSet
    vertexSets = [f["vertexSet"] for f in features]
    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn)

    cr_gold, re_gold = [], []
    cr_pred, re_pred = [], []

    for step, batch in tqdm(enumerate(dataloader), desc='eval'):
        cr_gold.extend(batch["cr_clusters"])      # cr_clusters
        re_gold.extend(batch["re_triples"])
        
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device),
                  "spans": batch["spans"],
                  "anaphors": batch["anaphors"],
                  "syntax_graph": batch["syntax_graph"].to(args.device),}
        with torch.no_grad():
            outputs = model.inference(**inputs)      
            cr_pred.extend(outputs['cr_predictions'])
            re_pred.extend(outputs['re_predictions'])

    # cr_pred: 是一个列表, 每一个元素对应一份Document中的clusters 簇集合
    # clusters 簇集合中的每个元素是一个cluster, 表示属于同一个实体的mention集合
    # 每一个mention, 也就是一个数字，表示的是一个index，指向spans里面的一个mention

    # cr_gold: 也是类似的，已经从事实span表中，转换到预测span表，每一个数字指向spans里面的一个mention
    # 如果在预测出的span中没有对应的，也就是预测错误或没有预测出，用-1来表示/占位
    
    # re_pred: 是一个列表，每一个元素对应一份Document中的triples 三元组集合{"h": i, "r": r, "t": j}
    # i 和 j是cluster的index，r是关系的index
    # re_gold: 也是类似的, 但它是没有经过转换的，其中的"h" "t"都是cr_gold.clusters的index
    cr_f1, cr_p, cr_r = compute_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold) # hard f1
    avg, muc, b3, ceafe = compute_avg_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold) # avg f1
    # get cluster mapping
    cluster_mappings = []
    for pc, gc in zip(cr_pred, cr_gold):
        cluster_mappings.append(get_cluster_mapping(pc, gc)[1])
    if tag != "test":
        re_f1, re_p, re_r, re_ign = compute_re_f1(cluster_mappings, re_pred, re_gold, vertexSets)
    else:
        re_f1, re_p, re_r, re_ign = .0, .0, .0, .0
    output_logs = {tag + "_cr_f1": cr_f1 * 100, tag + "_cr_p": cr_p * 100, tag + "_cr_r": cr_r * 100,
                   tag + "_re_f1": re_f1 * 100, tag + "_re_p": re_p * 100, tag + "_re_r": re_r * 100, tag + "_re_ign_f1": re_ign * 100,
                   tag + "_avg_f1": avg * 100, tag + "_muc_f1": muc * 100, tag + "_b3_f1": b3 * 100, tag + "_ceafe_f1": ceafe * 100}
    return re_f1, output_logs

def analyze_report(args, model: Table2Graph, features):
    model.eval()
    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn)

    cr_gold, re_gold = [], []
    cr_pred, re_pred = [], []

    for step, batch in tqdm(enumerate(dataloader), desc='eval'):
        cr_gold.extend(batch["cr_clusters"])      # cr_clusters
        re_gold.extend(batch["re_triples"])
        
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device),
                  "spans": batch["spans"],
                  "anaphors": batch["anaphors"],
                  "syntax_graph": batch["syntax_graph"].to(args.device),}
        with torch.no_grad():
            outputs = model.inference(**inputs)      
            cr_pred.extend(outputs['cr_predictions'])
            re_pred.extend(outputs['re_predictions'])
    
    report_list = []
    for feature, clusters_pred, clusters_gold, rels_pred, rels_gold in zip(features, cr_pred, cr_gold, re_pred, re_gold):
        report_dict = {}
        # 计算一个样例的coref tp, fp, fn
        cr_tp, cluster_map = get_cluster_mapping(clusters_pred, clusters_gold)
        cr_fp = len(clusters_pred) - cr_tp
        cr_fn = len(clusters_gold) - cr_tp
        report_dict['cr_tp'] = cr_tp
        report_dict['cr_fp'] = cr_fp
        report_dict['cr_fn'] = cr_fn
        # 计算这个样本中有多少簇是一定无法被预测出来的（span 没有划出来, 存在值为-1的index）
        cnt_cluster_could_not_be_predicted = 0 
        for cluster in clusters_gold:
            if -1 in cluster:
                cnt_cluster_could_not_be_predicted += 1
        report_dict['cnt_cluster_could_not_be_predicted'] = cnt_cluster_could_not_be_predicted
        # 分错的fp减去无法被预测的，就是模型本身在coref中的错误
        
        # relation
        # 计算这个样本中，有多少个relation是一定无法被预测出来的
        # 1. 因为提及没有被识别出来
        # 2. 在提及被识别出来的情况下，因为coref没有预测出正确的cluster
        reverse_cluster_map = get_reverse_cluster_map(clusters_pred, clusters_gold)
        cnt_rel_could_not_be_predicted_me = 0
        cnt_rel_could_not_be_predicted_coref = 0
        for rel in rels_gold:
            head_cluster = clusters_gold[rel['h']]
            tail_cluster = clusters_gold[rel['t']]
            # 如果head或者tail的cluster中有-1，那么这个relation是无法被预测出来的，此时原因为提及没有识别出来
            if -1 in head_cluster or -1 in tail_cluster:
                cnt_rel_could_not_be_predicted_me += 1
            # 又如果head或者tail没有在预测出的cluster找到能与之对应的，此时原因为coref没有预测出正确的cluster
            elif reverse_cluster_map[rel['h']] == -1 or reverse_cluster_map[rel['t']] == -1:
                cnt_rel_could_not_be_predicted_coref += 1
        report_dict['cnt_rel_could_not_be_predicted_me'] = cnt_rel_could_not_be_predicted_me
        report_dict['cnt_rel_could_not_be_predicted_coref'] = cnt_rel_could_not_be_predicted_coref

        # 计算tp
        # 将预测出的cluster映射到gold cluster上，然后计算relation的tp, fp, fn
        new_rels_pred = [{'h': cluster_map[r['h']], 't': cluster_map[r['t']], 'r': r['r']} for r in rels_pred]
        re_tp = 0
        for pr in new_rels_pred:
            if pr in rels_gold:
                re_tp += 1
        re_fp = len(rels_pred) - re_tp
        re_fn = len(rels_gold) - re_tp
        report_dict['re_tp'] = re_tp
        report_dict['re_fp'] = re_fp
        report_dict['re_fn'] = re_fn
    
        # 复原句子，可视化结果
        report_dict['raw_spans'] = feature['raw_spans']
        report_dict['sents'] = feature['sents']
        report_dict['labels'] = feature['labels']
        report_dict['clusters_pred'] = clusters_pred
        report_dict['clusters_gold'] = clusters_gold
        report_dict['cluster_map'] = cluster_map
        report_dict['reverse_cluster_map'] = reverse_cluster_map
        report_dict['new_rels_pred'] = new_rels_pred
        report_dict['rels_pred'] = rels_pred
        report_dict['rels_gold'] = rels_gold
        report_dict['num_anaphors'] = len(feature['anaphors'])
        
        report_list.append(report_dict)
    return report_list

def calculate_dict_list(report_list):
    # 计算tp, fp, fn
    cr_tp, cr_fp, cr_fn = 0, 0, 0
    re_tp, re_fp, re_fn = 0, 0, 0
    total_cluster_could_not_be_predicted = 0
    total_rel_could_not_be_predicted_me = 0
    total_rel_could_not_be_predicted_coref = 0
    for i in report_list:
        cr_tp += i['cr_tp']
        cr_fp += i['cr_fp']
        cr_fn += i['cr_fn']
        re_tp += i['re_tp']
        re_fp += i['re_fp']
        re_fn += i['re_fn']
        total_cluster_could_not_be_predicted += i['cnt_cluster_could_not_be_predicted']
        total_rel_could_not_be_predicted_me += i['cnt_rel_could_not_be_predicted_me']
        total_rel_could_not_be_predicted_coref += i['cnt_rel_could_not_be_predicted_coref']
    cr_p = cr_tp / (cr_tp + cr_fp + 1e-7)
    cr_r = cr_tp / (cr_tp + cr_fn + 1e-7)
    cr_f1 = (2 * cr_p * cr_r) / (cr_r + cr_p + 1e-7)
    re_p = re_tp / (re_tp + re_fp + 1e-7)
    re_r = re_tp / (re_tp + re_fn + 1e-7)
    re_f1 = (2 * re_p * re_r) / (re_r + re_p + 1e-7)
    return_dict = {}
    return_dict ['cr_tp'] = cr_tp
    return_dict ['cr_fp'] = cr_fp
    return_dict ['cr_fn'] = cr_fn
    return_dict ['cr_p'] = cr_p
    return_dict ['cr_r'] = cr_r
    return_dict ['cr_f1'] = cr_f1
    return_dict ['re_tp'] = re_tp
    return_dict ['re_fn'] = re_fn
    return_dict ['re_p'] = re_p
    return_dict ['re_p'] = re_p
    return_dict ['re_r'] = re_r
    return_dict ['re_f1'] = re_f1
    return_dict['total_cluster_could_not_be_predicted'] = total_cluster_could_not_be_predicted
    return_dict['total_rel_could_not_be_predicted_me'] = total_rel_could_not_be_predicted_me
    return_dict['total_rel_could_not_be_predicted_coref'] = total_rel_could_not_be_predicted_coref
    return return_dict

def report(args, model: Table2Graph, features):
    model.eval()
    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn)

    cr_gold = []
    cr_pred, re_pred = [], []

    for step, batch in tqdm(enumerate(dataloader)):
        cr_gold.extend(batch["cr_clusters"])      # cr_clusters
        
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device),
                  "spans": batch["spans"],
                  "anaphors": batch["anaphors"],
                  "syntax_graph": batch["syntax_graph"].to(args.device),}
        with torch.no_grad():
            outputs = model.inference(**inputs)      
            cr_pred.extend(outputs['cr_predictions'])
            re_pred.extend(outputs['re_predictions'])

    cr_f1, cr_p, cr_r = compute_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold) # hard f1
    avg, muc, b3, ceafe = compute_avg_cr_f1(pred_clusters=cr_pred, gold_clusters=cr_gold) # avg f1
    tag = "test"
    output_logs = {tag + "_cr_f1": cr_f1 * 100, tag + "_cr_p": cr_p * 100, tag + "_cr_r": cr_r * 100,
                   tag + "_avg_f1": avg * 100, tag + "_muc_f1": muc * 100, tag + "_b3_f1": b3 * 100, tag + "_ceafe_f1": ceafe * 100}
    print(output_logs)
    # get cluster mapping
    cluster_mappings = []
    for pc, gc in zip(cr_pred, cr_gold):
        cluster_mappings.append(get_cluster_mapping(pc, gc)[1])
    # get result
    result = []
    for cluster_mapping, pr, feature in zip(cluster_mappings, re_pred, features):
        for triple in pr:
            result.append(
                {
                    'title': feature['title'],
                    'h_idx': cluster_mapping[triple['h']],
                    't_idx': cluster_mapping[triple['t']],
                    'r': docred_id2rel[triple['r']],
                }
            )
    return result

def collate_fn(batch):
    """
    Reference: https://github.com/wzhouad/ATLOP/
    """
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    spans = [f["spans"] for f in batch]
    anaphors = [f["anaphors"] for f in batch]
    cr_clusters = [f["cr_clusters"] for f in batch]
    re_triples = [f["re_triples"] for f in batch]

    syntax_graph = [torch.tensor(f["syntax_graph"], dtype=float) for f in batch]
    syntax_graph = torch.block_diag(*syntax_graph)

    if batch[0]["cr_label"] is None:
        cr_label = None
        re_label = None
        re_table_label = None
        cr_table_label = None
        hts = None
        hts_table = None
    elif batch[0]["re_label"] is None:
        cr_label = []
        for f in batch:
            cr_label.extend(f["cr_label"])
        cr_label = torch.tensor(cr_label)
        re_label = None
        re_table_label = None
        cr_table_label = None
        hts = None
        hts_table = None
    else:
        cr_label = []
        for f in batch:
            cr_label.extend(f["cr_label"])
        cr_label = torch.tensor(cr_label)
        re_label = [torch.tensor(f["re_label"]) for f in batch] # in tensor form
        re_label = torch.cat(re_label, dim=0) 
        hts = [f["hts"] for f in batch]
        hts_table = [f["hts_table"] for f in batch]
        
        cr_table_label = []
        for f in batch:
            cr_table_label.extend(f["cr_table_label"])
        cr_table_label = torch.tensor(cr_table_label)
        re_table_label = []
        for f in batch:
            re_table_label.extend(f["re_table_label"])
        re_table_label = torch.tensor(re_table_label)

    output = {"input_ids": input_ids, "attention_mask": attention_mask, 
                "spans": spans, "hts": hts,
                "hts_table": hts_table,
                "cr_label": cr_label, "cr_clusters": cr_clusters,
                "re_label": re_label, "re_triples": re_triples,
                "syntax_graph": syntax_graph,
                "anaphors": anaphors,
                "cr_table_label": cr_table_label, "re_table_label": re_table_label,
                }
    return output

if __name__ == "__main__":

    args = get_opt()
    print(args)
    wandb.init(project="TaG", notes=args.notes)
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    # get config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config.num_class = len(docred_rel2id)
    config.num_gcn_layers = args.num_gcn_layers
    config.num_axial_layers = args.num_axial_layers
    config.alpha = args.alpha
    config.beta = args.beta
    config.rho = args.rho
    if config.model_type == "bert":
        bos_token, eos_token, pad_token = "[CLS]", "[SEP]", "[PAD]"
    elif config.model_type == "roberta":
        bos_token, eos_token, pad_token = "<s>", "</s>", "<pad>"
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = \
        bos_token_id, eos_token_id, pad_token_id

    config.dataset = args.dataset
    print(config)
    # get model
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    model = Table2Graph(config, encoder)
    model.to(device)

    if args.load_path == "":
        train_features = []
        if args.curriculum_pre25 != None and args.curriculum_pre50 != None and args.curriculum_pre75 != None:
            train_features.append(
                read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='gc', curriculum_threshold=args.curriculum_pre75,no_anaphor=args.no_anaphor)
            )
            train_features.append(
                read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='gc', curriculum_threshold=args.curriculum_pre50,no_anaphor=args.no_anaphor)
            )
            train_features.append(
                read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='gc', curriculum_threshold=args.curriculum_pre25,no_anaphor=args.no_anaphor)
            )
            train_features.append(
                read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='gc', curriculum_threshold=args.trust_threshold,no_anaphor=args.no_anaphor)
            )
        else:
            train_features.append(
                read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='gc', curriculum_threshold=args.trust_threshold,no_anaphor=args.no_anaphor)
            )
        dev_features = read_dataset(tokenizer, split='dev', dataset=args.dataset, task='gc',no_anaphor=args.no_anaphor)
        if args.dataset == "re-docred":
            test_features = read_dataset(tokenizer, split='test', dataset=args.dataset, task='gc',no_anaphor=args.no_anaphor)
        else:
            test_features = None

        train(args, model, train_features, dev_features, test_features)
    else:
        dev_features = read_dataset(tokenizer, split='dev', dataset=args.dataset, task='gc',no_anaphor=args.no_anaphor)
        test_features = read_dataset(tokenizer, split='test', dataset=args.dataset, task='gc',no_anaphor=args.no_anaphor)

        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        
        # ----------------my_analyze-------------
        # report_dict_list = analyze_report(args, model, dev_features)
        # result_dict = calculate_dict_list(report_dict_list)
        # print(f"total-------cr_p: {result_dict['cr_p']}, cr_r: {result_dict['cr_r']}, cr_f1: {result_dict['cr_f1']}------")
        # print(f"total-------re_p: {result_dict['re_p']}, re_r: {result_dict['re_r']}, re_f1: {result_dict['re_f1']}------")
        # print(f"total-------total_cluster_could_not_be_predicted: {result_dict['total_cluster_could_not_be_predicted']}------")
        # print(f"total-------total_rel_could_not_be_predicted_me: {result_dict['total_rel_could_not_be_predicted_me']}------")
        # print(f"total-------total_rel_could_not_be_predicted_coref: {result_dict['total_rel_could_not_be_predicted_coref']}------")
        # print(result_dict)
        # print("----------------------")
        
        # root_dir = os.path.dirname(os.path.realpath(__file__))
        # root_dir = os.path.dirname(root_dir)
        # if args.report_save_folder != "":
        #     if not os.path.isdir(args.report_save_folder):
        #         os.makedirs(args.report_save_folder)
        #     root_dir = args.report_save_folder
        # report_dir = os.path.join(root_dir, f"{args.dataset}")
        # if not os.path.isdir(report_dir):
        #     os.makedirs(report_dir)

        # # save as csv
        # import csv
        # file_path = os.path.join(report_dir, "report.csv")
        # fieldnames = set().union(*(d.keys() for d in report_dict_list))
        # # 使用字典的键作为CSV的列头
        # with open(file_path, 'w', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     writer.writerows(report_dict_list)

        # # save as json
        # file_path = os.path.join(report_dir, "report.json")
        # with open(file_path, "w") as f:
        #     json.dump(report_dict_list, f)
        # ----------------my_analyze-------------
        
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print("--------dev_output--------")
        print(dev_output)

        result = report(args, model, test_features)
        if args.dataset == "re-docred":
            _, test_output = evaluate(args, model, test_features, tag="re-docred-test")
            print("--------test_output--------")
            print(test_output)

        root_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(root_dir)
        result_dir = os.path.join(root_dir, f"result/{args.dataset}/gc/{args.notes}")
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(result, f)