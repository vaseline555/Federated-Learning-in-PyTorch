import torch

from transformers import SqueezeBertModel, SqueezeBertConfig



class SqueezeBert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, dropout, use_pt_model=False):
        super(SqueezeBert, self).__init__()
        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = SqueezeBertModel.from_pretrained('squeezebert/squeezebert-uncased')
            self.num_classes = num_classes
            self.num_embeddings = self.features.config.vocab_size
            self.embedding_size = self.features.config.embedding_size
            self.num_hiddens = self.features.config.hidden_size
            self.dropout = self.features.config.hidden_dropout_prob 
            
            self.classifier = torch.nn.Linear(self.features.config.hidden_size, self.num_classes, bias=True)

        else: # from scratch
            assert embedding_size == hidden_size, 'If you want embedding_size != intermediate hidden_size, please insert a Conv1d layer to adjust the number of channels before the first SqueezeBertModule.'
            self.num_classes = num_classes
            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_size
            self.num_hiddens = hidden_size
            self.dropout = dropout

            config = SqueezeBertConfig(
                vocab_size=self.num_embeddings,
                embedding_size=self.embedding_size,
                hidden_size=self.num_hiddens,
                hidden_dropout_prob=self.dropout
            )
            self.features = SqueezeBertModel(config)
            self.classifier = torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)['pooler_output']
        x = self.classifier(x)
        return x
