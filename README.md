## Requirements
* Python 3.6.8
* PyTorch 1.1.0
* pytorch-crf 0.7.2
* allennlp 1.0.0
* Transformers https://github.com/huggingface/transformers
* CUDA 9.0

### For BERT Embedding
Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface. Then change the value of parameter `--bert_model_dir` to the directory of the bert model.


