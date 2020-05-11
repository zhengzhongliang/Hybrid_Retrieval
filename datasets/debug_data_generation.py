import squad_retrieval
import sys
from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer

parent_folder_path = str(Path('.').absolute().parent)
sys.path+=[parent_folder_path]


def check_squad(check_raw_data = False, check_dataloader = False):
    # The raw retrieval look good.

    # The number of samples/sentences are not very consistent with what are reported in the paper, but I do not have a very elegant way to fix this.
    # Plus, such abnormal examples are very rare (about 100 in 100,000), so I will just leave them there.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _ = squad_retrieval.convert_squad_to_retrieval(tokenizer)

    if check_raw_data:
        squad_retrieval.check_squad_retrieval_pickle()

    if check_dataloader:
        squad_retrieval.check_squad_dataloader()


check_squad(check_raw_data=False, check_dataloader=True)

