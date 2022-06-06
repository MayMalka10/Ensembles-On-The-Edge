import pytorch_lightning as pl
import torch
from vector_quantize_pytorch import VectorQuantize
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.Metrics import PrecisionAtRecall, RecallAtPrecision
from kmeans_pytorch import kmeans
from sklearn.cluster._kmeans import _init_centroids
class SiameseEncoder( pl.LightningModule ):
    def __init__(self, encoder_decoder, primary_loss, n_embed=1024, decay=0.8, commitment=1, skip_quant=False,
                 skip_query_quant=False, learning_rate=1e-4, init_style='rand', **kwargs):
        super().__init__(**kwargs)

        self.encoder_decoder = encoder_decoder
        self.val_acc_handlers = {}
        self.val_acc_handlers["P@R=0.95"] = PrecisionAtRecall(recall_point=0.95)
        self.val_acc_handlers["R@P=0.95"] = RecallAtPrecision( precision_point=0.95 )
        self.train_acc_handler = PrecisionAtRecall(recall_point=0.95)
        self.n_embed = n_embed
        self.decay = decay
        self.commitment_w = commitment
        self.hparams.update(dict(n_embed=1024, decay=0.8, commitment_w=1., eps=1e-5))
        self.primary_loss = primary_loss
        self.skip_quant = skip_quant
        self.skip_query_quant = skip_query_quant
        self.learning_rate = learning_rate
        dummy_input = torch.zeros( (2, 1, 32, 32), device=self.device )
        self.quant_dim = self.encoder_decoder.encode( dummy_input ).shape[1]
        self.quantizer = VectorQuantize(
            dim=self.quant_dim,
            codebook_size=self.n_embed,     # size of the dictionary
            decay=self.decay,       # the exponential moving average decay, lower means the dictionary will change faster
            commitment=1.0,    # the weight on the commitment loss (==1 cause we want control)
            eps = 1e-5
        )
        self.init_style = init_style



        # pl.metrics.Precision
        # for k in range( len( self.decoder ) + 1 ):
        #     self.train_acc_handlers.append( pl.metrics.classification.iou.IoU( num_classes=n_classes ) )
        #     self.val_acc_handlers.append( pl.metrics.classification.iou.IoU( num_classes=n_classes ) )

    def init_vectors(self):
        max_elements = self.n_embed * 100
        embeddings = []
        n_elements = 0
        for batch in self.train_dataloader():
            embeddings.append( self.encoder_decoder.encode( batch[0].to( self.device ) ).data.detach() )
            n_elements += batch[0].shape[0]
            if n_elements > max_elements:
                break
        embeddings = torch.vstack( embeddings ).squeeze()
        if self.init_style == 'kmeans':
            cluster_ids_x, cluster_centers = kmeans(
                X=embeddings.reshape(-1,embeddings.shape[-1]), num_clusters=self.n_embed, distance='euclidean', device=self.device, tol=1e-2 )
            centers_torch = cluster_centers.T
        elif self.init_style == 'k_init':
            emb_np = embeddings.cpu().numpy()
            cluster_centers = _init_centroids( emb_np.reshape(-1,emb_np.shape[-1]), self.n_embed )
            centers_torch = torch.from_numpy( cluster_centers.T )
        elif self.init_style == 'simple':
            emb_np = embeddings.cpu().numpy()
            cluster_centers = _init_centroids( emb_np.reshape(-1,emb_np.shape[-1]), self.n_embed, init='random')
            centers_torch = torch.from_numpy( cluster_centers.T )
        self.quantizer.embed.data.copy_( centers_torch.data )
        self.quantizer.embed_avg.data.copy_( centers_torch.data )
        print('quantizer updated with new centroids!')

        # quantize, embed_ind, loss = self.quantizer(embeddings)
        # bins = torch.bincount(embed_ind)
        # print(f"after initialization, max_bin:{bins.max()}, min_bin:{bins.min()}")

    def normalize_embeddings(self, x, dim):
        eps = 1e-4
        xn = torch.norm( x, p=2, dim=dim).detach().unsqueeze(dim=dim)
        x = x.div(xn + eps)
        return x

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.init_style != 'rand':
            self.init_vectors()
        else:
            normalized_vectors = self.normalize_embeddings(self.quantizer.embed.data, dim=0)
            self.quantizer.embed.data.copy_( normalized_vectors.data )
            self.quantizer.embed_avg.data.copy_( normalized_vectors.data )

        if self.train_acc_handler:
            self.train_acc_handler.to(self.device)
        if self.val_acc_handlers:
            for handler in self.val_acc_handlers.values():
                handler.to(self.device)

    def calculate_acc(self, dists, labels, handler):
        return handler( dists, labels )
        # accs = []
        # for preds, handler in zip( preds_list, handlers[0:len( preds_list )] ):
        #     accs.append( handler( activation( preds ), gts ) )
        # accs.append( handlers[-1]( activation( ensemble_preds ), gts ) )
        # return accs

    def encode(self, x, skip_quant):
        z_e = self.encoder_decoder.encode( x )
        z_e = z_e.view( (z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]) )
        if not skip_quant:
            z_q, indices, commit_loss = self.quantizer( z_e )
        else:
            z_q, indices, commit_loss = z_e, None, 0
        z_q = z_q.view( (z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]) )
        return z_q, indices, commit_loss

    def decode(self, x):
        return self.encoder_decoder.decode(x)

    def process_batch(self, batch):
        data_a, data_p, label = batch
        emb_a, indices_a, commit_loss_a = self.encode( data_a, skip_quant=self.skip_query_quant)
        emb_p, indices_p, commit_loss_p = self.encode( data_p, skip_quant=self.skip_quant)
        out_a = self.decode(emb_a)
        out_p = self.decode( emb_p )
        prime_loss = self.primary_loss( out_a[label.squeeze() == 1], out_p[label.squeeze() == 1] )
        commit_loss = (commit_loss_a + commit_loss_p) / 2
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        if dists.dim() == 2:
            dists = dists.mean(dim=1)
        return prime_loss, commit_loss, label, dists

    def training_step(self, batch, batch_idx):
        prime_loss, commit_loss, labels, dists = self.process_batch( batch )
        if self.train_acc_handler:
            precision = self.calculate_acc( dists, labels, self.train_acc_handler )
            self.log( 'train_P@R', precision )
        self.log( 'prime_loss', prime_loss )
        self.log( 'commit_loss', commit_loss )
        return {'loss':(1-self.commitment_w) * prime_loss + commit_loss * self.commitment_w}

    def training_epoch_end(self, outputs_dicts) -> None:
        loss_sum = 0
        for output in outputs_dicts:
            loss_sum += output['loss']
        self.log( 'train_loss_epoch', loss_sum / len( outputs_dicts ) )
        if self.train_acc_handler:
            acc = self.train_acc_handler.compute()
            self.log( 'train_acc_epoch', acc )

    def validation_step(self, batch, batch_idx):
        prime_loss, commit_loss, labels, dists = self.process_batch( batch )
        acc_dict = {}
        if self.val_acc_handlers:
            for k in self.val_acc_handlers.keys():
                metric = self.calculate_acc(dists, labels, self.val_acc_handlers[k])
                # self.log(k, metric)
                acc_dict[k] = metric
        self.log( 'val_prime_loss', prime_loss )
        self.log( 'val_commit_loss', commit_loss )
        return {'loss':(1-self.commitment_w) * prime_loss + commit_loss * self.commitment_w}

    def validation_epoch_end(self, outputs_dicts) -> None:
        loss_sum = 0
        cnt = 0
        for output in outputs_dicts:
            loss_sum += output['loss']
        self.log( 'val_loss_epoch', loss_sum / len( outputs_dicts ) )
        if self.val_acc_handlers:
            acc_dict = {}
            for key, handler in self.val_acc_handlers.items():
                acc = handler.compute()
                acc_dict[key] = acc
                # self.logger.experiment.add_scalars( 'acc_epoch', acc_dict,
                #                                     global_step=self.global_step )
                self.log(f'{key}_val_epoch', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR( optimizer, T_max=300),
                'monitor': 'prime_loss',
            }
        }