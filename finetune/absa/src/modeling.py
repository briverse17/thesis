from transformers import AutoModel #, AutoTokenizer
from torch import nn, sigmoid

class ABSAClassifier(nn.Module):

    def __init__(self, pretrained_path, dropout, n_classes):
        super(ABSAClassifier, self).__init__()
        self.pretrained = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.pretrained.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        # Just get the embedding of the first (the "[CLS]") token
        x = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask
        )["last_hidden_state"][:, 0, :]
        # print(x.shape)
        x = self.dropout(x)
        x = self.linear(x)

        return sigmoid(x)
    
# # TEST
# params = {
#     "model": "briverse/vi-electra-small-cased",
#     "dropout": 0.02,
# }

# toker = AutoTokenizer.from_pretrained("briverse/vi-electra-small-cased")
# line = "Phòng rất bẩn . Tường đã bị ẩm mốc !"#, "View đẹp lắm mọi người !", "Đồ ăn chán"]
# encoding = toker.encode_plus(line, padding="max_length", max_length=128, return_tensors="pt")
# print(encoding["input_ids"].shape)
# model = ABSAClassifier(params, 102)
# output = model(
#     encoding["input_ids"],
#     encoding["attention_mask"]
# )
# print(toker.convert_ids_to_tokens(encoding["input_ids"][0]))
# print(output)
# print(output.shape)