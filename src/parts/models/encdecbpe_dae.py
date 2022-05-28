from nemo.collections.asr.models import EncDecCTCModelBPE
from omegaconf import DictConfig

class EncDecCTCModelBPE_dae(EncDecCTCModelBPE):
    def __init__(self, cfg: DictConfig, trainer=None, corruption_prob=0, use_dae=False, use_bsm=False):
        super().__init__(cfg=cfg, trainer=trainer)
        self.corruption_prob = corruption_prob
        
    def training_step(self, batch, batch_idx):
    
        ### Biderectional sequence modeling
        if use_bsm:
            flipped_batch = flip_batch(batch)

            new_batch = []

            new_batch.append(torch.vstack((batch[0], flipped_batch[0])))
            new_batch.append(torch.hstack((batch[1], flipped_batch[1])))
            new_batch.append(torch.vstack((batch[2], flipped_batch[2])))
            new_batch.append(torch.hstack((batch[3], flipped_batch[3])))
            batch = new_batch
        
        ###
        
        return super().training_step(batch, batch_idx)
    
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        
        ### Denoising auto-encoder
        if use_dae:
            for i in range(processed_signal.shape[0]):
                if i // 2:
                    corruption = np.random.choice([0,1], (processed_signal[i].shape[0], processed_signal[i].shape[1]),
                                                  p=[self.corruption_prob, 1 - self.corruption_prob])
                    processed_signal[i] = processed_signal[i].cpu() * corruption
                    processed_signal[i].cuda()
        
        ###
        
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions
