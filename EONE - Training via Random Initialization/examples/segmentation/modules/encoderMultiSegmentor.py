from quantizers import EncoderMultiDecoder
import pytorch_lightning as pl
import torch
from utils import visualizations
from segmentation_models_pytorch.utils.base import Activation


class EncoderMultiSegmentor(EncoderMultiDecoder.EncoderMultiDecoder):
    def  __init__(self, encoder, decoder, primary_loss, n_classes, **kwargs):
        super().__init__( encoder, decoder, primary_loss, **kwargs)

        self.train_acc_handlers = []
        self.val_acc_handlers = []
        for k in range(len(self.decoder) + 1):
            self.train_acc_handlers.append(pl.metrics.classification.iou.IoU(num_classes=n_classes))
            self.val_acc_handlers.append(pl.metrics.classification.iou.IoU(num_classes=n_classes))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        for handler in self.train_acc_handlers + self.val_acc_handlers:
            handler.to(self.device)

    def ensemble_calculator(self, preds_list):
        return torch.mean( torch.stack( preds_list ), axis=0 )

    def training_step(self, batch, batch_idx):
        result_dict = super().training_step(batch, batch_idx)
        # accs = self.calculate_acc(result_dict['preds'], result_dict['gts'], self.train_acc_handlers)
        # acc_dict = {}
        # for idx in range( len( self.decoder ) ):
        #     acc_dict[f'model_{idx}_train'] = accs[idx]
        # acc_dict["ensemble_train"] = accs[-1]
        # self.logger.experiment.add_scalars('acc_step', acc_dict,
        #                                     global_step=self.global_step )
        return result_dict

    def training_epoch_end(self, outputs_dicts) -> None:
        super().training_epoch_end(outputs_dicts)
        acc_dict = {}
        for idx in range( len( self.decoder ) ):
            acc_dict[f'model_{idx}_train'] = self.train_acc_handlers[idx].compute()
        acc_dict["ensemble_train"] = self.train_acc_handlers[-1].compute()
        self.logger.experiment.add_scalars( 'acc_epoch', acc_dict,
                                            global_step=self.global_step )


    def validation_step(self, batch, batch_idx):
        result_dict = super().validation_step( batch, batch_idx )
        accs = self.calculate_acc(result_dict['preds'], result_dict['gts'], self.val_acc_handlers, Activation('softmax2d'))
        acc_dict = {}
        for idx in range( len( self.decoder ) ):
            acc_dict[f'model_{idx}_val'] = accs[idx]
        acc_dict["ensemble_val"] = accs[-1]
        self.logger.experiment.add_scalars('acc_step', acc_dict,
                                            global_step=self.global_step )
        if not batch_idx % 10:
            fig_gt, ax_gt = visualizations.overlay_mask_on_image(batch[0][0], result_dict['gts'][0].squeeze())
            self.logger.experiment.add_figure('image_and_gtmask', fig_gt, global_step=self.global_step)
            for idx in range(len(result_dict['preds'])):
                fig_pred, ax_pred = visualizations.overlay_mask_on_image(batch[0][0], result_dict['preds'][idx][0].max(dim=0)[1])
                ax_pred.text(1, 10, f"acc:{accs[idx].item():.2f}", fontsize=12, backgroundcolor='w')
                self.logger.experiment.add_figure(f'image_and_mask_model_{idx}', fig_pred, global_step=self.global_step)
        return result_dict



    def validation_epoch_end(self, outputs_dicts) -> None:
        super().validation_epoch_end( outputs_dicts )
        # acc_dict = {}
        # for idx in range( len( self.decoder ) ):
        #     acc_dict[f'model_{idx}_val'] = self.val_acc_handlers[idx].compute()
        # acc_dict["ensemble_val"] = self.val_acc_handlers[-1].compute()
        # self.logger.experiment.add_scalars('acc_epoch', acc_dict,
        #                                    global_step=self.global_step)
