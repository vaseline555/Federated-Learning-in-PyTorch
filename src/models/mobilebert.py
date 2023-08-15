import torch

from transformers import MobileBertModel, MobileBertConfig



class MobileBert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, dropout, use_pt_model, is_seq2seq):
        super(MobileBert, self).__init__()
        self.is_seq2seq = is_seq2seq
        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = MobileBertModel.from_pretrained('google/mobilebert-uncased')
            self.num_classes = num_classes
            self.num_embeddings = self.features.config.vocab_size
            self.embedding_size = self.features.config.embedding_size
            self.num_hiddens = self.features.config.hidden_size
            self.dropout = self.features.config.hidden_dropout_prob 

            self.classifier = torch.nn.Linear(self.features.config.hidden_size, self.num_classes, bias=True)

        else: # from scratch
            self.num_classes = num_classes
            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_size
            self.num_hiddens = hidden_size
            self.dropout = dropout

            config = MobileBertConfig(
                vocab_size=self.num_embeddings,
                embedding_size=self.embedding_size,
                hidden_size=self.num_hiddens,
                hidden_dropout_prob=self.dropout
            )
            self.features = MobileBertModel(config)
            self.classifier = torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x['last_hidden_state'] if self.is_seq2seq else x['pooler_output'])
        return x
