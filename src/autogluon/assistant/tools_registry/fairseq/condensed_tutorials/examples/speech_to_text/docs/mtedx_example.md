# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing speech translation (ST) systems using the Multilingual TEDx dataset. It covers data preparation with feature extraction and vocabulary generation, ASR and ST model implementation using fairseq, and techniques for both single-language and multilingual models. Key functionalities include training transformer-based models with encoder pretraining, checkpoint averaging, multilingual training with language ID tokens, and evaluation using WER/BLEU metrics. The tutorial provides complete command-line workflows for training, inference, and evaluation, with special attention to multilingual model considerations like language token handling and performance comparisons between monolingual and multilingual approaches.

*This is a condensed version that preserves essential implementation details and context.*

# Speech Translation on Multilingual TEDx

This guide covers implementing speech translation (ST) using the Multilingual TEDx dataset, which contains TEDx talks in 8 source languages with translations to 5 target languages.

## Data Preparation

```bash
# Install required packages
pip install pandas torchaudio soundfile sentencepiece

# Generate TSV manifests, features, vocabulary for each language
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task asr \
  --vocab-type unigram --vocab-size 1000
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st \
  --vocab-type unigram --vocab-size 1000

# Add vocabulary and configuration for joint data
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task asr --joint \
  --vocab-type unigram --vocab-size 8000
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st --joint \
  --vocab-type unigram --vocab-size 8000
```

## ASR Implementation

### Training

Single language (Spanish example):
```bash
fairseq-train ${MTEDX_ROOT}/es-es \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset valid_asr \
    --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_xs --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER} \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10
```

Joint model (all 8 languages):
```bash
fairseq-train ${MTEDX_ROOT} \
    --config-yaml config_asr.yaml \
    --train-subset train_es-es_asr,train_fr-fr_asr,train_pt-pt_asr,train_it-it_asr,train_ru-ru_asr,train_el-el_asr,train_ar-ar_asr,train_de-de_asr \
    --valid-subset valid_es-es_asr,valid_fr-fr_asr,valid_pt-pt_asr,valid_it-it_asr,valid_ru-ru_asr,valid_el-el_asr,valid_ar-ar_asr,valid_de-de_asr \
    --save-dir ${MULTILINGUAL_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10 \
    --ignore-prefix-size 1
```

**Important**: 
- `--update-freq 8` simulates 8 GPUs with 1 GPU
- For multilingual models, use `--ignore-prefix-size 1` to exclude target language ID token from training loss

### Inference & Evaluation

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Generate and evaluate
fairseq-generate ${MTEDX_ROOT}/es-es \
  --config-yaml config_asr.yaml --gen-subset test --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --skip-invalid-size-inputs-valid-test \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct --remove-bpe
```

For multilingual models:
```bash
# Evaluate each language
for LANG in es fr pt it ru el ar de; do
  fairseq-generate ${MTEDX_ROOT} \
    --config-yaml config_asr.yaml --gen-subset test_${LANG}-${LANG}_asr --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct --remove-bpe
done
```

## ST Implementation

### Training

Bilingual example (Es-En):
```bash
fairseq-train ${MTEDX_ROOT}/es-en \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset valid_st \
    --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_xs --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER} \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10
```

Multilingual model (all 12 directions):
```bash
fairseq-train ${MTEDX_ROOT} \
    --config-yaml config_st.yaml \
    --train-subset train_el-en_st,train_es-en_st,train_es-fr_st,train_es-it_st,train_es-pt_st,train_fr-en_st,train_fr-es_st,train_fr-pt_st,train_it-en_st,train_it-es_st,train_pt-en_st,train_pt-es_st,train_ru-en_st \
    --valid-subset valid_el-en_st,valid_es-en_st,valid_es-fr_st,valid_es-it_st,valid_es-pt_st,valid_fr-en_st,valid_fr-es_st,valid_fr-pt_st,valid_it-en_st,valid_it-es_st,valid_pt-en_st,valid_pt-es_st,valid_ru-en_st \
    --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10 \
    --ignore-prefix-size 1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER}
```

**Best practices**:
- Pre-train the ST encoder with ASR for faster training and better performance
- Use `--ignore-prefix-size 1` for multilingual models to exclude target language ID token from loss

### Inference & Evaluation

```bash
# Average checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Generate and evaluate
fairseq-generate ${MTEDX_ROOT}/es-en \
  --config-yaml config_st.yaml --gen-subset test --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu --remove-bpe
```

For multilingual models:
```bash
for LANGPAIR in es-en es-fr es-pt fr-en fr-es fr-pt pt-en pt-es it-en it-es ru-en el-en; do
  fairseq-generate ${MTEDX_ROOT} \
    --config-yaml config_st.yaml --gen-subset test_${LANGPAIR}_st --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --scoring sacrebleu --remove-bpe
done
```

**Important**: For multilingual models, use `--prefix-size 1` to force decoding from the target language ID token.

## Performance Results

### ASR Results (WER)
| Model | Params | Es | Fr | Pt | It | Ru | El | Ar | De |
|-------|--------|----|----|----|----|----|----|----|----|
| Monolingual | 10M | 46.4 | 45.6 | 54.8 | 48.0 | 74.7 | 109.5 | 104.4 | 111.1 |

### ST Results (BLEU)
| Model | Params | Es-En | Es-Pt | Es-Fr | Fr-En | Fr-Es | Fr-Pt | Pt-En | Pt-Es | It-En | It-Es | Ru-En | El-En |
|-------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Bilingual | 10M | 7.0 | 12.2 | 1.7 | 8.9 | 10.6 | 7.9 | 8.1 | 8.7 | 6.4 | 1.0 | 0.7 | 0.6 |
| Multilingual | 31M | 12.3 | 17.4 | 6.1 | 12.0 | 13.6 | 13.2 | 12.0 | 13.7 | 10.7 | 13.1 | 0.6 | 0.8 |