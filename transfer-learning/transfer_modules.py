from libraries import *

import random

def on_train_start(models):
    for model in models:
        model.configure_optimizers()
        model.configure_callbacks()
        model.on_fit_start()
        model.setup("fit")
        model.on_pretrain_routine_start()
        model.on_pretrain_routine_end()
        model.on_train_start()
    
def on_epoch_start(models):
    for model in models:
        model.on_epoch_start()
        model.on_train_epoch_start()
    
def on_epoch_end(models):
    for model in models:
        model.on_train_epoch_end() 
        model.on_epoch_end()
    
def on_train_end(models):
    for model in models:
        model.on_train_end()
        model.on_fit_end()
        model.teardown("fit")    

def flip_batch(batch):
    for i in range(len(batch)):
        try:
            for j in range(len(batch[i])):
                batch[i][j] = torch.flip(batch[i][j], dims=[0])
        except Exception:
            pass
    return batch


def paired_asr_training_step(asr_model, batch, batch_idx, epoch, dataloader_idx=0):
    batch = asr_model.on_before_batch_transfer(batch, dataloader_idx)
    batch = asr_model.transfer_batch_to_device(batch, torch.device("cuda"), dataloader_idx)
    batch = asr_model.on_after_batch_transfer(batch, dataloader_idx)
    
    signal, signal_len, transcript, transcript_len = batch
    
    for i in range(signal.shape[0] // 2):
        for j in range(signal.shape[1]):
            if random.random() < 0.3:
                signal[i][j] = 0
    
    asr_model.on_train_batch_start(batch, batch_idx, dataloader_idx)
    
    asr_model.on_before_zero_grad(asr_model._optimizer)
    asr_model.optimizer_zero_grad(epoch, 0, asr_model._optimizer, batch_idx)
    
    log_probs, encoded_len, predictions = asr_model.forward(input_signal=signal, input_signal_length=signal_len)
    
    loss = asr_model.loss(
        log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
    )
    
    if loss == 0:
        loss += 10e+3
    
    del log_probs, encoded_len, predictions
#     del signal, signal_len, transcript, transcript_len
    
    return loss

def paired_tts_training_step(spec_gen, vocoder, batch, batch_idx, epoch, dataloader_idx=0, flag=False):      
    spec_gen.on_train_batch_start(batch, batch_idx, dataloader_idx)
    
    spec_gen.on_before_zero_grad(spec_gen._optimizer)
    spec_gen.optimizer_zero_grad(epoch, 0, spec_gen._optimizer, batch_idx)

    batch = spec_gen.on_before_batch_transfer(batch, dataloader_idx)
    batch = spec_gen.transfer_batch_to_device(batch, torch.device("cuda"), dataloader_idx)
    batch = spec_gen.on_after_batch_transfer(batch, dataloader_idx)
    
    signal, signal_len, transcript, transcript_len, attn_prior, pitch, speaker = batch

    for i in range(transcript.shape[0] // 2):
        for j in range(transcript.shape[1]):
            if random.random() < 0.3:
                transcript[i][j] = 0
    
    # |-------------------------------------------------------------|

    attn_prior, durs, speaker = None, None, None
    audio, audio_lens, text, text_lens, attn_prior, pitch, speaker = batch

    mels, spec_len = spec_gen.preprocessor(input_signal=audio, length=audio_lens)

    mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch = spec_gen(
        text=text,
        durs=durs,
        pitch=pitch,
        speaker=speaker,
        pace=1.0,
        spec=mels,
        attn_prior=attn_prior,
        mel_lens=spec_len,
        input_lens=text_lens,
    )
    if durs is None:
        durs = attn_hard_dur

    mel_loss = spec_gen.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
    dur_loss = spec_gen.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
    loss = mel_loss + dur_loss
#         if self.learn_alignment:
    ctc_loss = spec_gen.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
    bin_loss_weight = min(spec_gen.current_epoch / spec_gen.bin_loss_warmup_epochs, 1.0) * 1.0
    bin_loss = spec_gen.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
    loss += ctc_loss + bin_loss

    if pitch is not None:
        pitch_loss = spec_gen.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss += pitch_loss
        del pitch_loss

    # |-------------------------------------------------------------|

    # |-------------------------------------------------------------|

    batch = [audio, audio_lens, mels_pred]

    vocoder.on_train_batch_start(batch, batch_idx, dataloader_idx)

    batch = vocoder.on_before_batch_transfer(batch, dataloader_idx)
    batch = vocoder.transfer_batch_to_device(batch, torch.device("cuda:1"), dataloader_idx)
    batch = vocoder.on_after_batch_transfer(batch, dataloader_idx)

    # |-------------------------------------------------------------|

    audio, audio_len, audio_mel = batch

    # Mel as input for L1 mel loss
    audio_trg_mel, _ = vocoder.trg_melspec_fn(audio, audio_len)
    audio = audio.unsqueeze(1)

    audio_pred = vocoder.generator(x=audio_mel)
    audio_pred_mel, _ = vocoder.trg_melspec_fn(audio_pred.squeeze(1), audio_len)

    # Train discriminator

    if True:
        loss_d = torch.tensor(0)
    else:
        optim_g, optim_d = vocoder.optim_g, vocoder.optim_g

        vocoder.on_before_zero_grad(optim_d)
        vocoder.optimizer_zero_grad(epoch, batch_idx, optim_d, batch_idx)

        mpd_score_real, mpd_score_gen, _, _ = vocoder.mpd(y=audio, y_hat=audio_pred.detach())
        loss_disc_mpd, _, _ = vocoder.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = vocoder.msd(y=audio, y_hat=audio_pred.detach())
        loss_disc_msd, _, _ = vocoder.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd

    # |-------------------------------------------------------------|

#         del audio, audio_lens, text, text_lens, attn_prior, pitch, speaker
#         del mel_loss, dur_loss, ctc_loss, bin_loss_weight, bin_loss
#         del audio_trg_mel, audio_pred, audio_pred_mel
#         del mpd_score_real, mpd_score_gen, msd_score_real, msd_score_gen
    
    return loss.cpu(), loss_d.cpu()

def dt_text_training_step(asr_model, spec_gen, vocoder, batch, batch_idx, epoch, dataloader_idx=0):    
    batch = asr_model.on_before_batch_transfer(batch, dataloader_idx)
    batch = asr_model.transfer_batch_to_device(batch, torch.device("cuda"), dataloader_idx)
    batch = asr_model.on_after_batch_transfer(batch, dataloader_idx)
    
    _, _, text, text_lens = batch
    
    target = asr_model._wer.ctc_decoder_predictions_tensor(
        text, predictions_len=text_lens, return_hypotheses=False,
    )

    parsed = spec_gen.parse(target[0])
        
    mels_pred = spec_gen.generate_spectrogram(tokens=parsed)
    signal = vocoder.convert_spectrogram_to_audio(spec=mels_pred.cuda(1))
    signal_len = torch.tensor([len(signal[0])])
        
    del mels_pred, target, parsed
    
    batch = [signal, signal_len, text, text_lens]
    
    return paired_asr_training_step(asr_model, batch, batch_idx, epoch, dataloader_idx=0)

def dt_signal_training_step(asr_model, spec_gen, vocoder, batch, batch_idx, epoch, dataloader_idx=0, flag=True):    
    batch = spec_gen.on_before_batch_transfer(batch, dataloader_idx)
    batch = spec_gen.transfer_batch_to_device(batch, torch.device("cuda"), dataloader_idx)
    batch = spec_gen.on_after_batch_transfer(batch, dataloader_idx)

    signal, signal_len, _, _ = batch
    
    log_probs, encoded_len, predictions = asr_model.forward(input_signal=signal, input_signal_length=signal_len)

    current_hypotheses = asr_model._wer.ctc_decoder_predictions_tensor(
        predictions, predictions_len=encoded_len, return_hypotheses=False,
    )
    
    try:
        transcript = spec_gen.parse(current_hypotheses[0])
    except:
        transcript = torch.tensor([[0, 0]]).cuda(0)

    transcript_len = torch.tensor([len(transcript)])
        
    batch = (signal.cpu().detach(), signal_len.cpu().detach(), transcript.cpu().detach(), transcript_len, None, None, None)
    
    del log_probs, encoded_len, predictions, current_hypotheses

    return paired_tts_training_step(spec_gen, vocoder, batch, batch_idx, epoch, dataloader_idx=0)

def asr_training_step(asr_model, spec_gen, vocoder, asr_batch, raw_signal_batch, batch_idx, epoch, dataloader_idx=0, dt=True):
    asr_loss = 0
    
    asr_loss += paired_asr_training_step(asr_model, asr_batch, batch_idx, epoch=epoch)
    asr_loss += paired_asr_training_step(asr_model, flip_batch(asr_batch), batch_idx, epoch=epoch)
    if dt:
        try:
            asr_loss += dt_text_training_step(asr_model, spec_gen, vocoder, raw_signal_batch, batch_idx, epoch)
            asr_loss += dt_text_training_step(asr_model, spec_gen, vocoder, flip_batch(raw_signal_batch), batch_idx, epoch)
        except:
            pass
            print('DT text error')

    asr_loss.cuda()

    asr_model.on_before_backward(asr_loss)
    asr_model.backward(asr_loss, asr_model._optimizer, batch_idx)
    asr_model.on_after_backward()

    asr_model.on_before_optimizer_step(asr_model._optimizer, batch_idx)
    asr_model.optimizer_step(epoch, 0, asr_model._optimizer, batch_idx)

#     asr_model.on_train_batch_end(predictions, asr_batch, batch_idx, dataloader_idx)
    
    return asr_loss.cpu().detach().numpy()

def tts_training_step(asr_model, spec_gen, vocoder, tts_batch, raw_signal_batch, batch_idx, epoch, dataloader_idx=0, dt=True):
    fastpitch_loss, hifigan_loss = 0, 0
    
    loss, loss_d = paired_tts_training_step(spec_gen, vocoder, tts_batch, batch_idx, epoch=epoch, flag=True)
    fastpitch_loss, hifigan_loss = fastpitch_loss + loss, hifigan_loss + loss_d
    del loss, loss_d
    loss, loss_d = paired_tts_training_step(spec_gen, vocoder, flip_batch(tts_batch), batch_idx, epoch=epoch, flag=True)
    fastpitch_loss, hifigan_loss = fastpitch_loss + loss, hifigan_loss + loss_d
    del loss, loss_d
        
    if dt:
        loss, loss_d = dt_signal_training_step(asr_model, spec_gen, vocoder, raw_signal_batch, batch_idx, epoch, flag=True)
        fastpitch_loss, hifigan_loss = fastpitch_loss + loss, hifigan_loss + loss_d
        del loss, loss_d
        loss, loss_d = dt_signal_training_step(asr_model, spec_gen, vocoder, flip_batch(raw_signal_batch), batch_idx, epoch, flag=True)
        fastpitch_loss, hifigan_loss = fastpitch_loss + loss, hifigan_loss + loss_d
        del loss, loss_d
    
    fastpitch_loss.cuda()
    hifigan_loss.cuda(1)
    
    spec_gen.on_before_backward(fastpitch_loss)
    spec_gen.backward(fastpitch_loss, spec_gen._optimizer, batch_idx)
    spec_gen.on_after_backward()

    spec_gen.on_before_optimizer_step(spec_gen._optimizer, batch_idx)
    spec_gen.optimizer_step(epoch, 0, spec_gen._optimizer, batch_idx)

    if hifigan_loss > 0:

        vocoder.on_before_backward(hifigan_loss)
        vocoder.backward(hifigan_loss, vocoder.optim_d, 0)
        vocoder.on_after_backward()

        vocoder.on_before_optimizer_step(vocoder.optim_d, batch_idx)
        vocoder.optimizer_step(epoch, batch_idx, vocoder.optim_d, 0)
        vocoder.scheduler_d.step()
        
    return fastpitch_loss.cpu().detach().numpy(), hifigan_loss.cpu().detach().numpy()
    
#     vocoder.on_train_batch_end(audio_pred, batch, batch_idx, dataloader_idx)
#     spec_gen.on_train_batch_end(mels_pred, batch, batch_idx, dataloader_idx)
    
def asr_val(model, dataloader):
    torch.set_grad_enabled(False)
    model.on_validation_start()
    model.on_epoch_start()
    model.on_validation_epoch_start()

    val_outs = []

    for batch_idx, val_batch in enumerate(dataloader):

        model.on_validation_batch_start(val_batch, batch_idx, 0)

        batch = model.on_before_batch_transfer(val_batch, 0)
        batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
        batch = model.on_after_batch_transfer(batch, 0)

        out = model.cuda().validation_step(batch, batch_idx)

        val_loss, val_wer_num, val_wer_denom, val_wer = out
        
        model.on_validation_batch_end(out, batch, batch_idx, 0)
        val_outs.append(out)
        
    model.on_epoch_end()
    model.on_validation_end()

    torch.set_grad_enabled(True)
    
    val_wer = []
    for item in val_outs:
        val_wer.append(item['val_wer'])

    return torch.mean(torch.tensor(val_wer))
    
def tts_val(model, dataloader):
    
    losses = []
    
    for batch_idx, batch in enumerate(dataloader):
        
        model.on_validation_batch_start(batch, batch_idx, 0)

        batch = model.on_before_batch_transfer(batch, 0)
        batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
        batch = model.on_after_batch_transfer(batch, 0)
        
        out = model.validation_step(batch, batch_idx)

        model.on_validation_batch_end(out, batch, batch_idx, 0)
        
        losses.append(out['val_loss'].cpu().detach().numpy())
    return np.mean(losses)