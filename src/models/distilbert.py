import torch

from transformers import DistilBertModel, DistilBertConfig



class DistilBert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, dropout, use_pt_model=False):
        super(DistilBert, self).__init__()
        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.num_classes = num_classes
            self.num_embeddings = self.features.config.vocab_size
            self.embedding_size = self.features.config.dim
            self.num_hiddens = self.features.config.hidden_size
            self.dropout = self.features.config.dropout

            self.classifier = torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)

        else: # from scratch
            self.num_classes = num_classes
            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_size
            self.num_hiddens = hidden_size
            self.dropout = dropout

            config = DistilBertConfig(
                vocab_size=self.num_embeddings,
                dim=self.embedding_size,
                hidden_size=self.num_hiddens,
                hidden_dropout_prob=self.dropout
            )
            self.features = DistilBertModel(config)
            self.classifier = torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)[0]
        x = self.classifier(x[:, 0, :]) # use [CLS] token for classification
        return x
    