mkdir data # make data folder

mkdir -p data/mscoco_imgfeat
mkdir -p data/msmoco_imgfeat/train2014
mkdir -p data/mscoco_imgfeat/val2014
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip

mkdir -p data/questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P data/
unzip data/v2_Questions_Train_mscoco.zip -d data/questions && rm data/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P data/
unzip data/v2_Questions_Val_mscoco.zip -d data/questions && rm data/v2_Questions_Val_mscoco.zip

mkdir -p data/annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P data/
unzip data/v2_Annotations_Train_mscoco.zip -d data/annotations && rm data/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P data/
unzip data/v2_Annotations_Val_mscoco.zip -d data/annotations && rm  data/v2_Annotations_Val_mscoco.zip

# get glove embeddings
mkdir -p data/glove
wget https://nlp.stanford.edu/data/glove.6B.zip -P data
unzip data/glove.6B.zip -d data/glove && rm data/glove.6B.zip
rm data/glove/glove.6B.200d.txt data/glove/glove.6B.100d.txt data/glove/glove.6B.50d.txt

# convert tsv tp numpy
python src/data/tsv_to_npy.py -p  data/mscoco_imgfeat/train2014_obj36.zip -s data/mscoco_imgfeat/train2014
python src/data/tsv_to_npy.py -p  data/mscoco_imgfeat/val2014_obj36.zip -s data/mscoco_imgfeat/val2014
