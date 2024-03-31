import torch
import pytorch_lightning as pl
from diffusion.diffusion_helpers import GaussianFourierProjection

from diffusion.diffusion_loss import DiffusionLossMetric

from .scheduler import CosineWarmupScheduler
from ponita.models.ponita import PonitaFiberBundle
from ponita.transforms.random_rotate import RandomRotate


fourier_scale = 16
t_emb_dim = 64
class PONITA_DIFFUSION(pl.LightningModule):
    """
    """

    def __init__(self, args, num_atomic_states: int):
        super().__init__()

        # Store some of the relevant args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup
        if args.layer_scale == 0.:
            args.layer_scale = None

        # For rotation augmentations during training and testing
        self.train_augm = args.train_augm
        self.rotation_transform = RandomRotate(['pos', 'L0'], n=3) # TODO: I'm not sure if we can rotate each of these matricies like this. Maybe it works?
        # Note I'm not rotating the fractional coords "X0", since these are lengths

        self.t_emb = GaussianFourierProjection(
            t_emb_dim // 2, fourier_scale
        )

        # The metrics to log
        self.train_metric = DiffusionLossMetric(args.num_timesteps)
        self.valid_metric = DiffusionLossMetric(args.num_timesteps)
        self.test_metric = DiffusionLossMetric(args.num_timesteps)

        # Input/output specifications:
        in_channels_scalar = num_atomic_states + 64 # atomic_number + the time embedding
        in_channels_vec = 0 # since the position is already encoded in the graph
        out_channels_scalar = num_atomic_states # atomic_number
        out_channels_vec = 1  # The cartesian_pos score (gradient of where the atom should be in the next step)

        # Make the model
        self.model = PonitaFiberBundle(in_channels_scalar + in_channels_vec,
                        args.hidden_dim,
                        out_channels_scalar,
                        args.layers,
                        output_dim_vec = out_channels_vec,
                        radius=args.radius,
                        num_ori=args.num_ori,
                        basis_dim=args.basis_dim,
                        degree=args.degree,
                        widening_factor=args.widening_factor,
                        layer_scale=args.layer_scale,
                        task_level='node',
                        multiple_readouts=args.multiple_readouts)
        # should we have lift_graph=True???

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, graph):
        if self.train_augm:
            # TODO: fix this. because L0's dimension is NOT the same as the number of atoms, this rotate_transform function will not work
            # graph = self.rotation_transform(graph)
            pass

        loss = self.train_metric.update(self, graph, self.t_emb)
        print("loss", loss["loss"])
        return loss

    def on_train_epoch_end(self):
        self.log("train MAE (energy)", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pass
        # pred_energy, pred_force = self.pred_energy_and_force(graph)
        # self.valid_metric(pred_energy * self.scale + self.shift, graph.energy)
        # self.valid_metric_force(pred_force * self.scale, graph.force)        

    def on_validation_epoch_end(self):
        self.log("valid MAE", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pos_pred = self(graph)
        self.test_metric(pos_pred, graph.y)  

    def on_test_epoch_end(self):
        self.log("test MSE", self.test_metric)

    def configure_optimizers(self):
        """
        Adapted from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('layer_scale'):
                    no_decay.add(fpn)
                elif pn.endswith('gaussian_fourier_proj_w'):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}