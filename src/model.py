import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from modules.table_filler import BaseTableFiller, get_hrt, gen_hts, convert_node_to_table
from modules.mention_extraction import SequenceTagger
from modules.coreference_resolution import CoreferenceResolutionTableFiller
from modules.relation_extraction import RelationExtractionTableFiller
from modules.graph import RelGCN
from long_seq import process_long_input, process_multiple_segments

num_ner = 6
        
class MEModel(nn.Module):
    """
    Solely compute mention spans, and store in advance.
    """
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config # include custom config (args), basically extended from BertConfig
        self.bert = encoder
        
        self.hidden_size = config.hidden_size
        self.tagger = SequenceTagger(self.hidden_size)

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output

    def compute_loss(self, input_ids=None, attention_mask=None, label=None):
        sequence_output = self.encode(input_ids, attention_mask)
        loss1 = self.tagger.compute_loss(sequence_output, attention_mask, label)
        return loss1

    def inference(self, input_ids=None, attention_mask=None):
        sequence_output = self.encode(input_ids, attention_mask)
        preds = self.tagger.inference(sequence_output, attention_mask)
        return preds

class JointTableModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()        
        self.config = config
        self.bert = encoder

        self.hidden_size = config.hidden_size
        self.CR = CoreferenceResolutionTableFiller(self.hidden_size)
        self.RE = RelationExtractionTableFiller(self.hidden_size, config.num_class, beta=config.beta)
        self.alpha = config.alpha
        self.beta = config.beta

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output, attention

    def forward(self, input_ids=None, attention_mask=None):
        # return self.compute_loss(input_ids, attention_mask, mr_labels)
        return

    def compute_loss(self, input_ids=None, attention_mask=None, spans=None, 
                     hts=None, cr_label=None, re_label=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        cr_loss = self.CR.compute_loss(hs, ts, cr_label)
        re_loss = self.RE.compute_loss(hs, ts, re_label)
        return self.alpha * cr_loss + re_loss

    def inference(self, input_ids=None, attention_mask=None, spans=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        cr_predictions = self.CR.inference(hs, ts, span_len, hts)
        re_predictions = self.RE.inference(hs, ts, span_len, hts, cr_predictions)
        outputs = {'cr_predictions': cr_predictions, 're_predictions': re_predictions}
        return outputs

class Table2Graph(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()        
        self.config = config
        self.bert = encoder

        self.hidden_size = config.hidden_size

        self.span_attention = nn.MultiheadAttention(self.hidden_size, num_heads=1, batch_first=True)

        self.CRTablePredictor = BaseTableFiller(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                                block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RETablePredictor = BaseTableFiller(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                                block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RGCN = RelGCN(self.hidden_size, self.hidden_size, self.hidden_size, num_rel=3, num_layer=config.num_gcn_layers)

        self.CR = CoreferenceResolutionTableFiller(self.hidden_size)
        self.RE = RelationExtractionTableFiller(self.hidden_size, config.num_class, beta=config.beta)

        self.coref_contrastive_loss = CorefContrastiveLoss()

        self.alpha = config.alpha
        self.beta = config.beta
        self.rho = config.rho

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output, attention

    def forward(self, input_ids=None, attention_mask=None, spans=None, 
                    anaphors=None, syntax_graph=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        span_len = [len(span) for span in spans]
        anaphor_len = [len(anaphor) for anaphor in anaphors]
        hts = [gen_hts(l+a_l) for l, a_l in zip(span_len, anaphor_len)]
        hs, ts, rs = self.get_hrt_span_anaphor(sequence_output, attention, spans, anaphors, hts)

        cr_table, _ = self.CRTablePredictor.forward(hs, ts)
        re_table, _ = self.RETablePredictor.forward(hs, ts)

        # convert logits to tabel in batch form
        offset = 0
        cr_adj, re_adj = [], [] # store sub table first
        for l, a_l in zip(span_len, anaphor_len):
            cr_sub = cr_table[offset: offset + (l+a_l)*(l+a_l)].view((l+a_l), (l+a_l))
            re_sub = re_table[offset: offset + (l+a_l)*(l+a_l)].view((l+a_l), (l+a_l))
            cr_sub = torch.softmax(cr_sub, dim=-1)
            re_sub = torch.softmax(re_sub, dim=-1)
            cr_adj.append(cr_sub)
            re_adj.append(re_sub)
            offset += (l+a_l)*(l+a_l)
        cr_adj = torch.block_diag(*cr_adj)
        re_adj = torch.block_diag(*re_adj)
        sg_adj = syntax_graph

        adjacency_list = [cr_adj, re_adj, sg_adj]
        nodes = self.get_node_embed_s_and_a(sequence_output, spans, anaphors)

        nodes = self.RGCN(nodes, adjacency_list)
        nodes = self.remove_anaphor_node(nodes, span_len, anaphor_len)

        hs, ts = convert_node_to_table(nodes, span_len)
        hs = torch.cat([hs, rs], dim=-1)
        ts = torch.cat([ts, rs], dim=-1)

        cr_logits, _ = self.CR.forward(hs, ts)
        re_logits, _ = self.RE.forward(hs, ts)

        return cr_logits, re_logits

    def compute_loss(self, input_ids=None, attention_mask=None, spans=None, 
                     hts_table=None, cr_label=None, re_label=None,
                     cr_table_label=None, re_table_label=None,
                     anaphors=None, syntax_graph=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, ts, rs = self.get_hrt_span_anaphor(sequence_output, attention, spans, anaphors, hts_table)

        # graph structure prediction & compute auxiliary loss
        cr_table_loss, (cr_table_h_l, cr_table_h_bl), cr_table = self.CRTablePredictor.compute_loss(hs, ts, cr_table_label, return_logit=True)
        re_table_loss, (re_table_h_l, re_table_h_bl), re_table = self.RETablePredictor.compute_loss(hs, ts, re_table_label, return_logit=True)
        constrative_hidden = torch.cat([cr_table_h_l, cr_table_h_bl], dim=-1)
        cr_table_contrastive_loss = self.compute_coref_contrastive_loss(constrative_hidden, cr_table_label, spans, anaphors)

        # convert logits to table in batch form
        span_len = [len(span) for span in spans]
        anaphor_len = [len(anaphor) for anaphor in anaphors]
        offset = 0
        cr_adj, re_adj = [], [] # store sub table first
        for l, a_l in zip(span_len, anaphor_len):
            cr_sub = cr_table[offset: offset + (l+a_l)*(l+a_l)].view((l+a_l), (l+a_l))
            re_sub = re_table[offset: offset + (l+a_l)*(l+a_l)].view((l+a_l), (l+a_l))
            cr_sub = torch.softmax(cr_sub, dim=-1)
            re_sub = torch.softmax(re_sub, dim=-1)
            cr_adj.append(cr_sub)
            re_adj.append(re_sub)
            offset += (l+a_l)*(l+a_l)
        cr_adj = torch.block_diag(*cr_adj)
        re_adj = torch.block_diag(*re_adj)
        sg_adj = syntax_graph

        adjacency_list = [cr_adj, re_adj, sg_adj]
        nodes = self.get_node_embed_s_and_a(sequence_output, spans, anaphors)

        nodes = self.RGCN(nodes, adjacency_list)
        nodes = self.remove_anaphor_node(nodes, span_len, anaphor_len)

        hs, ts = convert_node_to_table(nodes, span_len)
        hs = torch.cat([hs, rs], dim=-1)
        ts = torch.cat([ts, rs], dim=-1)

        cr_loss, (cr_h_l, cr_h_bl) = self.CR.compute_loss(hs, ts, cr_label)
        re_loss, (re_h_l, re_h_bl) = self.RE.compute_loss(hs, ts, re_label)
        constrative_hidden = torch.cat([cr_h_l, cr_h_bl], dim=-1)
        cr_contrastive_loss = self.compute_coref_contrastive_loss(constrative_hidden, cr_label, spans)
 
        return cr_loss + re_loss + self.alpha * cr_table_loss + self.alpha * re_table_loss

    def inference(self, input_ids=None, attention_mask=None, spans=None, 
                    anaphors=None, syntax_graph=None):
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]

        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "spans": spans,
                  "anaphors": anaphors,
                  "syntax_graph": syntax_graph}

        cr_logits, re_logits = self.forward(**inputs) 
        cr_logits = cr_logits.to(dtype=torch.float64)
        cr_logits = torch.sigmoid(cr_logits)
        re_logits = re_logits.to(dtype=torch.float64)
        ####################################################
        # Levenstein decoding
        ####################################################
        re_labels = self.RE.get_label(re_logits)[:, 1:].bool()
        lev_reg = []
        offset = 0
        for batch_idx, l in enumerate(span_len):
            square = re_labels[offset:offset+l*l].view(l, l, -1)
            levmat = torch.zeros(l, l).to(re_labels.device)
            for i in range(l):
                for j in range(l):
                    lev_ij = (square[i, :]^square[j, :]).float().sum() + (square[:, i]^square[:, j]).float().sum()
                    levmat[i][j] = lev_ij
            levmat = torch.sigmoid(levmat)
            lev_reg.append(levmat.view(-1, 1))
            offset += l*l
        lev_reg = torch.cat(lev_reg, dim=0).to(dtype=torch.float64)
        cr_logits = cr_logits - self.rho * lev_reg
        ####################################################
        # decoding cr
        ####################################################
        cr_predictions = self.CR.inference(span_len=span_len, batch_hts=hts, logits=cr_logits)
        ####################################################
        # decoding re
        ####################################################
        re_predictions = self.RE.inference(span_len=span_len, batch_hts=hts, batch_clusters=cr_predictions, logits=re_logits)
        
        outputs = {'cr_predictions': cr_predictions, 're_predictions': re_predictions}
        return outputs

    def remove_anaphor_node(self, nodes, span_len, anaphor_len):
        offset = 0
        new_nodes = []
        for l, a_l in zip(span_len, anaphor_len):
            new_nodes.append(nodes[offset: offset + l])
            offset += l + a_l
        new_nodes = torch.cat(new_nodes, dim=0)
        return new_nodes

    def get_hrt_span_anaphor(self, sequence_output: torch.Tensor=None, attention: torch.Tensor=None,\
        batch_span_pos=None, batch_anaphors_pos=None, batch_hts=None, strategy='marker'):
        """
        span_pos, anaphors_pos and hts are in batch format.
        attention shape: (batch_size, num_heads, sequence_length, sequence_length).
        """
        hss, tss, rss, rss_span = [], [], [], []
        for i, (span_pos, anaphors_pos, hts) in enumerate(zip(batch_span_pos, batch_anaphors_pos, batch_hts)):
            span_hts = gen_hts(len(span_pos))
            both_atts= []
            span_embs = []
            anaphor_embs = []
            for span in span_pos:
                if strategy == 'marker':
                    emb = sequence_output[i, span[0]]
                    att = attention[i, :, span[0]]
                else:
                    raise ValueError("Unimplemented strategy.")
                span_embs.append(emb)
                both_atts.append(att)
            for anaphors in anaphors_pos:
                emb = sequence_output[i, anaphors]
                att = attention[i, :, anaphors]
                anaphor_embs.append(emb)
                both_atts.append(att)
            span_embs = torch.stack(span_embs, dim=0)
            if len(anaphor_embs) != 0:
                anaphor_embs = torch.stack(anaphor_embs, dim=0)
                key_value = span_embs.unsqueeze(0)
                query = anaphor_embs.unsqueeze(0)
                (out_attention, _) = self.span_attention(query=query, key=key_value, value=key_value)
                out_attention = out_attention.squeeze(0)
                anaphor_embs = (anaphor_embs + out_attention) / 2
                both_embs = torch.cat([span_embs, anaphor_embs], dim=0)   # [n_s + n_a, d]
            else:
                both_embs = span_embs
            both_atts = torch.stack(both_atts, dim=0)   # [n_s + n_a, num_heads, seq_len]

            hts = torch.LongTensor(hts).to(sequence_output.device)
            hs = torch.index_select(both_embs, 0, hts[:, 0])        # [(n_s+n_a)^2, d]
            ts = torch.index_select(both_embs, 0, hts[:, 1])        # [(n_s+n_a)^2, d]

            h_att = torch.index_select(both_atts, 0, hts[:, 0])    # [(n_s+n_a)^2, num_heads, seq_len]
            t_att = torch.index_select(both_atts, 0, hts[:, 1])    # [(n_s+n_a)^2, num_heads, seq_len]
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            span_hts = torch.LongTensor(span_hts).to(sequence_output.device)
            h_att_span = torch.index_select(both_atts, 0, span_hts[:, 0])    # [(n_s+n_a)^2, num_heads, seq_len]
            t_att_span = torch.index_select(both_atts, 0, span_hts[:, 1])    # [(n_s+n_a)^2, num_heads, seq_len]
            ht_att_span = (h_att_span * t_att_span).mean(1)
            ht_att_span = ht_att_span / (ht_att_span.sum(1, keepdim=True) + 1e-5)
            rs_span = contract("ld,rl->rd", sequence_output[i], ht_att_span)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            rss_span.append(rs_span)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        rss_span = torch.cat(rss_span, dim=0)
        hss = torch.cat([hss, rss], dim=-1)
        tss = torch.cat([tss, rss], dim=-1)
        return hss, tss, rss_span
    
    def get_node_embed_s_and_a(self, sequence_output: torch.Tensor=None, batch_span_pos=None, batch_anaphor_pos=None):
        """
        get node embed (including span embed and anaphor embed)
        """
        embs = []
        for i, (span_pos, anaphor_pos) in enumerate(zip(batch_span_pos, batch_anaphor_pos)):
            span_embs = []
            anaphor_embs = []
            for span in span_pos:
                emb = sequence_output[i, span[0]]
                span_embs.append(emb)
            for anaphor in anaphor_pos:
                emb = sequence_output[i, anaphor]
                anaphor_embs.append(emb)
            span_embs = torch.stack(span_embs, dim=0)
            if len(anaphor_pos) != 0:
                anaphor_embs = torch.stack(anaphor_embs, dim=0)
                key_value = span_embs.unsqueeze(0)
                query = anaphor_embs.unsqueeze(0)
                (out_attention, _) = self.span_attention(query=query, key=key_value, value=key_value)
                out_attention = out_attention.squeeze(0)
                anaphor_embs = (anaphor_embs + out_attention) / 2
                both_embs = torch.cat([span_embs, anaphor_embs], dim=0)
            else:
                both_embs = span_embs
            embs.append(both_embs)
        embs = torch.cat(embs, dim=0)
        return embs

    def compute_coref_contrastive_loss(self, batch_embs, cr_labels, spans, anaphors=None, temperature=0.2):
        """
        batch_embs: (num_mentions, hidden_dim)
        cr_labels:  (sum(num_mentions^2), )
        """
        label_list = []
        emb_list = []
        if anaphors == None:
            span_len = [len(span) for span in spans]
            offset = 0
            offset_emb = 0
            for l in span_len:
                label_per_sample = cr_labels[offset: offset + l*l]
                label_per_sample = label_per_sample.view(l, l)
                label_list.append(label_per_sample)
                emb = batch_embs[offset_emb : offset_emb + l]
                emb_list.append(emb)
                offset += l*l
                offset_emb += l
        else:
            span_len = [len(span) for span in spans]
            anaphor_len = [len(anaphor) for anaphor in anaphors]
            offset = 0
            offset_emb = 0
            for l, a_l in zip(span_len, anaphor_len):
                label_per_sample = cr_labels[offset: offset + (l+a_l)*(l+a_l)]
                label_per_sample = label_per_sample.view(l+a_l, l+a_l)
                label_list.append(label_per_sample)
                emb = batch_embs[offset_emb : offset_emb + l+a_l]
                emb_list.append(emb)
                offset += (l+a_l)*(l+a_l)
                offset_emb += l+a_l
        total_loss = 0
        for label, emb in zip(label_list, emb_list):
            loss = self.coref_contrastive_loss(emb, label, temperature)
            total_loss += loss
        total_loss = total_loss / len(label_list)
        return total_loss

def compute_coref_contrastive_loss_2(self, corf_logits, cr_labels, spans, anaphors=None, temperature=0.2):
        """
        corf_logits: (sum(num_mentions^2), dim)
        cr_labels:  (sum(num_mentions^2), )
        """
        label_list = []
        logits_list = []
        if anaphors == None:
            span_len = [len(span) for span in spans]
            offset = 0
            for l in span_len:
                label_per_sample = cr_labels[offset: offset + l*l]
                label_per_sample = label_per_sample.view(l, l)
                label_list.append(label_per_sample)
                logit = corf_logits[offset: offset + l*l]
                logit = logit.view(l, l)
                logits_list.append(logit)
                offset += l*l
        else:
            span_len = [len(span) for span in spans]
            anaphor_len = [len(anaphor) for anaphor in anaphors]
            offset = 0
            for l, a_l in zip(span_len, anaphor_len):
                label_per_sample = cr_labels[offset: offset + (l+a_l)*(l+a_l)]
                label_per_sample = label_per_sample.view(l+a_l, l+a_l)
                label_list.append(label_per_sample)
                logit = corf_logits[offset: offset + (l+a_l)*(l+a_l)]
                logit = logit.view(l, l)
                logits_list.append(logit)
                offset += (l+a_l)*(l+a_l)
        total_loss = 0
        for label, logit in zip(label_list, logits_list):
            """
            label: (num_mentions, num_mentions)
            logit: (sum(num_mentions^2), dim)
            """
            n = label.size(0)
            device = label.device
            label_matrix = torch.zeros(n**2, n**2, device=device)
            for a_index, line in enumerate(label):
                connect_list = [[a_index, a_index]]
                for b_index, value in enumerate(line):
                    if value == 1:
                        connect_list.append([a_index, b_index])
                for i in connect_list:
                    for j in connect_list:
                        label_matrix[i[0]*n+i[1], j[0]*n+j[1]] = 1
            loss = self.coref_contrastive_loss(logit, label_matrix, temperature)
            total_loss += loss
        total_loss = total_loss / len(label_list)
        return total_loss

class CorefContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, mention_embs, cr_labels, temperature=0.2):
        """
        mention_embs: (num_mentions, hidden_dim)
        cr_labels:  (num_mentions, num_mentions)
        """
        n = mention_embs.size(0)
        device = mention_embs.device
        mask = torch.eye(n, device=device).bool()
        # 对每一对mention计算余弦相似度
        cosine_sim_matrix = F.cosine_similarity(mention_embs.unsqueeze(1), mention_embs.unsqueeze(0), dim=2)
        cosine_sim_matrix = cosine_sim_matrix[~mask].reshape(n, n-1)
        cosine_sim_matrix = cosine_sim_matrix / temperature
        cosine_sim_matrix = self.log_softmax(cosine_sim_matrix)

        cr_labels = cr_labels[~mask].reshape(n, n-1)
        loss = - cr_labels * cosine_sim_matrix
        loss = loss.sum(dim=-1).mean()
        return loss
