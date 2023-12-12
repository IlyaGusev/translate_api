rm -f nllb-200_3.3B_int8_ct2.zip
rm -f flores200_sacrebleu_tokenizer_spm.model
wget https://pretrained-nmt-models.s3.us-west-2.amazonaws.com/CTranslate2/nllb/nllb-200_3.3B_int8_ct2.zip
wget https://pretrained-nmt-models.s3.us-west-2.amazonaws.com/CTranslate2/nllb/flores200_sacrebleu_tokenizer_spm.model
rm -rf models
mkdir -p models
mv nllb-200_3.3B_int8_ct2.zip models/ && cd models && unzip nllb-200_3.3B_int8_ct2.zip && cd ..
mv flores200_sacrebleu_tokenizer_spm.model models/
