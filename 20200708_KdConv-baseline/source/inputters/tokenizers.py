import torch
from transformers import BertTokenizer


class ModBertTokenizer:
    def __init__(self, model_type, pad_token="[PAD]", bos_token="<s>",
                 eos_token="</s>", unk_token="[UNK]", sep_token="[SEP]",
                 cls_token="[CLS]", mask_token="[MASK]", new_tokens=[], special_token_dict={}):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # load from pretrained tokenizer
        self.pretrained = BertTokenizer.from_pretrained(model_type)

        # add special tokens
        self._adapt_vocab(special_token_dict)
        self.pretrained.add_tokens([bos_token, eos_token])  # different from other apis
        self.pretrained.add_tokens(new_tokens)

        # vocab dict and revserse vocab dict
        self.word2id = self.pretrained.vocab
        self.word2id.update(self.pretrained.added_tokens_encoder)
        self.id2word = self.pretrained.ids_to_tokens
        self.id2word.update(self.pretrained.added_tokens_decoder)

        # set special token ids
        for token_type in ["pad_token", "bos_token", "eos_token",
                           "unk_token", "sep_token", "cls_token",
                           "mask_token"]:
            token = getattr(self, token_type)
            setattr(self, f"{token_type}_id", self.word2id[token])
            self.pretrained.add_special_tokens({token_type: getattr(self, token_type)})
        for token_type, token in special_token_dict.items():
            setattr(self, f"{token_type}_id", self.word2id[token])
            self.pretrained.add_special_tokens({token_type: getattr(self, token_type)})

        self.vocab_size = len(self)

    def __len__(self):
        return len(self.word2id)

    def _adapt_vocab(self, special_token_dict):
        self.pretrained.add_tokens(list(special_token_dict.values()))

    def encode(self, sent):
        if isinstance(sent, str):
            token_ids = self.pretrained.convert_tokens_to_ids(self.pretrained.tokenize(sent))
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            return token_ids
        else:
            return [self.encode(s) for s in sent]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            with torch.cuda.device_of(ids):
                ids = ids.tolist()
        if not isinstance(ids[0], list):
            tokens = self.pretrained.convert_ids_to_tokens(ids)
            if tokens[0] == self.bos_token:
                tokens = tokens[1:]
            sent = []
            for w in tokens:
                if w != self.eos_token:
                    sent.append(w)
                else:
                    break
            sent = [w for w in sent if w not in (self.pad_token,)]
            sent = self.pretrained.convert_tokens_to_string(sent)
            sent = sent.replace(' ', '')

            return sent
        else:
            return [self.decode(x) for x in ids]
