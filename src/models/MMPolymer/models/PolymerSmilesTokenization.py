# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from transformers import AddedToken, PreTrainedTokenizer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 本地分词器文件配置
_REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_TOKENIZER_DIR = os.environ.get(
    "POLYMER_TOKENIZER_DIR",
    str(_REPO_ROOT / "data" / "tokenizer"),
)
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class PolymerSmilesTokenizer(PreTrainedTokenizer):
    """
    基于 RoBERTa 的 SMILES 分词器，使用字节级 BPE 编码。
    
    参数:
        vocab_file: 词表文件路径
        merges_file: 合并规则文件路径
        errors: 解码字节时的错误处理方式，默认为 "replace"
        add_prefix_space: 是否在输入前添加空格，默认为 False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *init_inputs, **kwargs):
        """从本地目录加载分词器（忽略预训练模型名称和 HF 缓存）"""
        vocab_path = os.path.join(LOCAL_TOKENIZER_DIR, VOCAB_FILES_NAMES["vocab_file"])
        merges_path = os.path.join(LOCAL_TOKENIZER_DIR, VOCAB_FILES_NAMES["merges_file"])
        
        if not (os.path.isfile(vocab_path) and os.path.isfile(merges_path)):
            raise FileNotFoundError(
                f"找不到本地分词器文件: {vocab_path} 或 {merges_path}。"
                f"请将 vocab.json 和 merges.txt 放置在 {LOCAL_TOKENIZER_DIR}/ 目录下。"
            )
        
        return cls(vocab_file=vocab_path, merges_file=merges_path, **kwargs)

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs
    ):
        # 将字符串转换为 AddedToken
        def _ensure_added_token(token, lstrip=False, rstrip=False):
            return AddedToken(token, lstrip=lstrip, rstrip=rstrip) if isinstance(token, str) else token
        
        # 先加载词表和合并规则（必须在调用super().__init__之前）
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        with open(merges_file, encoding="utf-8") as f:
            bpe_merges = [tuple(merge.split()) for merge in f.read().split("\n")[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 初始化特殊tokens
        bos_token = _ensure_added_token(bos_token)
        eos_token = _ensure_added_token(eos_token)
        sep_token = _ensure_added_token(sep_token)
        cls_token = _ensure_added_token(cls_token)
        unk_token = _ensure_added_token(unk_token)
        pad_token = _ensure_added_token(pad_token)
        mask_token = _ensure_added_token(mask_token, lstrip=True)  # mask token 包含前置空格

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # 初始化编解码器
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        
        # SMILES 正则表达式模式
        smi_regex_pattern = r"(\-?[0-9]+\.?[0-9]*|\[|\]|SELF|Li|Be|Na|Mg|Al|K|Ca|Co|Zn|Ga|Ge|As|Se|Sn|Te|N|O|P|H|I|b|c|n|o|s|p|Br?|Cl?|Fe?|Ni?|Si?|\||\(|\)|\^|=|#|-|\+|\\|\/|@|\*|\.|\%|\$)"
        self.pat = re.compile(smi_regex_pattern)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """对文本进行分词"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 将字节映射为 unicode 字符串，避免 BPE 控制字符
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """将 token 转换为 ID"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """将 ID 转换为 token"""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """将 token 序列转换为字符串"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        构建模型输入序列，添加特殊标记。
        - 单序列: `<s> X </s>`
        - 双序列: `<s> A </s></s> B </s>`
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """获取特殊标记掩码：1 表示特殊标记，0 表示序列标记"""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """创建 token type IDs（RoBERTa 不使用，返回全零列表）"""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

def load_tokenizer() -> PolymerSmilesTokenizer:
    """
    加载本地分词器（便捷函数）
    
    返回:
        PolymerSmilesTokenizer: 已加载的分词器实例
    异常:
        FileNotFoundError: 当本地词表文件缺失时抛出
    """
    tokenizer = PolymerSmilesTokenizer.from_pretrained()
    logger.info("分词器加载成功")
    return tokenizer