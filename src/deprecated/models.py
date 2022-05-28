from libraries import *

config_path = 'stt_en_citrinet_256_gamma_0_25'
config_name = 'model_config.yaml'
yaml = YAML(typ='safe')

with open(os.path.join(config_path, config_name)) as f:
    config = yaml.load(f)

config['tokenizer']['dir'] = 'citrinet_tokenizer/tokenizer_spe_unigram_v1024'
config['tokenizer']['type'] = 'bpe'

# config['train_ds']['manifest_filepath']="../../datasets/LJSpeech-1.1/small_manifest.json"
config['train_ds']['manifest_filepath']="./an4/train_manifest_200.json"
config['train_ds']['batch_size'] = 1
config['train_ds']['num_workers'] = 4
config['train_ds']['pin_memory'] = True

# config['validation_ds']['manifest_filepath']="../../datasets/LJSpeech-1.1/train_manifest.json"
config['validation_ds']['manifest_filepath']="./an4/test_manifest.json"
config['validation_ds']['batch_size'] = 1
config['validation_ds']['num_workers'] = 4
config['validation_ds']['pin_memory'] = True

config['tokenizer']['model_path'] = 'stt_en_citrinet_256_gamma_0_25/3d20ebb793c84a64a20c7ad26fc64d62_tokenizer.model'
config['tokenizer']['vocab_path'] = 'stt_en_citrinet_256_gamma_0_25/df5191f216004f10a268c44e90fdb63f_vocab.txt'
config['tokenizer']['spe_tokenizer_vocab'] = 'stt_en_citrinet_256_gamma_0_25/b774eaac83804907843607272fde21a4_tokenizer.vocab'

# config['init_from_nemo_model'] = 'asr_model.nemo'

asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=DictConfig(config))
# asr_model.maybe_init_from_pretrained_checkpoint(cfg=DictConfig(config))


config_path = 'configs'
config_name = 'fastpitch_align.yaml'
yaml = YAML(typ='safe')

with open(os.path.join(config_path, config_name)) as f:
    config = yaml.load(f)

config['model']['train_ds']['manifest_filepath']="./an4/train_manifest_200.json"
config['model']['train_ds']['batch_size'] = 1
config['model']['train_ds']['num_workers'] = 4
config['model']['train_ds']['pin_memory'] = True

config['model']['validation_ds']['manifest_filepath']="./an4/test_manifest.json"
config['model']['validation_ds']['batch_size'] = 1
config['model']['validation_ds']['num_workers'] = 4
config['model']['validation_ds']['pin_memory'] = True

# config['init_from_nemo_model'] = './tts_en_fastpitch_align.nemo'

spec_gen = FastPitchModel.from_config_dict(DictConfig(config['model']))
# spec_gen.maybe_init_from_pretrained_checkpoint(cfg=DictConfig(config))

config_path = 'conf/hifigan'
config_name = 'hifigan22050.yaml'
yaml = YAML(typ='safe')

with open(os.path.join(config_path, config_name)) as f:
    config = yaml.load(f)

config['init_from_nemo_model'] = './tts_hifigan.nemo'
    
vocoder = HifiGanModel.from_config_dict(DictConfig(config))
vocoder.maybe_init_from_pretrained_checkpoint(cfg=DictConfig(config))
