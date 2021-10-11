mkdir data # make data folder

mkdir -p data/mscoco_imgfeat
mkdir -p data/mscoco_imgfeat/train2014
mkdir -p data/mscoco_imgfeat/val2014
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip

mkdir -p data/questions
wget -P data/questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json
wget -P data/questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json

mkdir -p data/annotations
wget -P data/annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json
wget -P data/annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json

# get glove embeddings
mkdir -p data/glove
wget https://nlp.stanford.edu/data/glove.6B.zip -P data
unzip data/glove.6B.zip -d data/glove && rm data/glove.6B.zip
rm data/glove/glove.6B.200d.txt data/glove/glove.6B.100d.txt data/glove/glove.6B.50d.txt

# download spacy en
pip install -U pip setuptools wheel
pip install -U spacy
python3 -m spacy download en
python3 -m spacy download en_core_web_sm

# convert tsv tp numpy
python3 src/data/tsv_to_npy.py -p  data/mscoco_imgfeat/train2014_obj36.tsv -s data/mscoco_imgfeat/train2014
python3 src/data/tsv_to_npy.py -p  data/mscoco_imgfeat/val2014_obj36.tsv -s data/mscoco_imgfeat/val2014

pip install -U wandb torchtext torchvision scipy