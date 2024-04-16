import torch
import torch.nn.functional as F


# https://dzdata.medium.com/intro-to-diffusion-model-part-3-5d699e5f0714
# This is the clearest form of the loss function. Everyone else just keeps a summation of terms from t=0 to t=tau.
# But that form is not workable unless you have fancy (unoptimized) logic in your dataloader.
#
# So I am using the loss from here (the official d3pm implementation) instead:
# https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
class D3pmLoss2(torch.nn.Module):
    def __init__(self, *, num_atomic_states: int):
        super().__init__()
        self.model_prediction = "x_start"  # x_start, xprev
        self.model_output = "logits"  # logits or logistic_pars # TODO(curtis): I don't understand logistic_pars. Maybe we DO need to use that. for now, I'll use logits
        # self.loss_type = loss_type  # kl, hybrid, cross_entropy_x_start
        self.hybrid_coeff = 0.2  # hybrid_coeff
        self.score_matching_coeff = 0.3  # TODO(curtis): tune this
        # self.jax_dtype = jax_dtype

        # Data \in {0, ..., num_atomic_states-1}
        self.num_atomic_states = num_atomic_states
        # self.transition_bands = transition_bands
        # self.transition_mat_type = transition_mat_type
        self.eps = data_diffusor.eps
        self.data_diffusor = data_diffusor

    # --------------------------------------
    # KL Divergence loss
    # --------------------------------------

    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """Computes logits for an underlying logistic distribution."""
        loc = loc.unsqueeze(-1)
        log_scale = log_scale.unsqueeze(-1)
        # Shift log_scale such that if it's zero the probs have a scale
        # that is not too wide and not too narrow either.
        inv_scale = torch.exp(-(log_scale - 2.0))
        bin_width = 2.0 / (self.num_atomic_states - 1.0)
        bin_centers = torch.linspace(
            start=-1.0, end=1.0, steps=self.num_atomic_states, device=loc.device
        )
        bin_centers = bin_centers.view(1, *([1] * (loc.ndim - 2)), -1)
        bin_centers = bin_centers - loc
        log_cdf_min = F.logsigmoid(inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = F.logsigmoid(inv_scale * (bin_centers + 0.5 * bin_width))
        logits = self.log_min_exp(log_cdf_plus, log_cdf_min, self.eps)
        # Normalization:
        # # Option 1:
        # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
        # # value x = 0.
        # logits[..., 0] = log_cdf_plus[..., 0]
        # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
        # # value x = 255.
        # log_one_minus_cdf_min = - F.softplus(inv_scale * (bin_centers - 0.5 * bin_width))
        # logits[..., -1] = log_one_minus_cdf_min[..., -1]
        # # Option 2:
        # # Alternatively normalize by reweighting all terms. This avoids
        # # sharp peaks at 0 and 255.
        # since we are outputting logits here, we don't need to do anything.
        # they will be normalized by softmax anyway.
        return logits

    def log_min_exp(self, a, b, eps):
        """Computes log(min(exp(a), exp(b))) in a numerically stable way."""
        return (
            torch.where(
                a < b,
                a + torch.log1p(torch.exp(b - a)),
                b + torch.log1p(torch.exp(a - b)),
            )
            + eps
        )

    def p_logits(self, *, x, t, model_output):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)
        # model_output = model_fn(x, t)
        if self.model_output == "logits":
            model_logits = model_output
        elif self.model_output == "logistic_pars":
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == "x_start":
            # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
            pred_x_start_logits = model_logits
            t_broadcast = t.view(t.shape + (1,) * (model_logits.ndim - 1))
            model_logits = torch.where(
                t_broadcast == 0,
                pred_x_start_logits,
                self.data_diffusor.q_posterior_logits(
                    pred_x_start_logits, x, t, x_start_logits=True
                ),
            )
        elif self.model_prediction == "xprev":
            # Use the logits out of the model directly as the logits for
            # p(x_{t-1}|x_t). model_logits are already set correctly.
            # NOTE: the pred_x_start_logits in this case makes no sense.
            # For Gaussian DDPM diffusion the model predicts the mean of
            # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
            # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
            # The equivalent for the categorical case is nontrivial.
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        assert (
            model_logits.shape
            == pred_x_start_logits.shape
            == x.shape + (self.num_atomic_states,)
        )
        return model_logits, pred_x_start_logits

    def categorical_kl_logits(self, logits1, logits2, eps=1.0e-6):
        """KL divergence between categorical distributions.
        Distributions parameterized by logits.
        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.
        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = F.softmax(logits1 + eps, dim=-1) * (
            F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1)
        )
        return torch.sum(out, dim=-1)

    def vb_terms_bpd(self, *, x_start, x_t, t, output_logits):
        """Calculate specified terms of the variational bound.
        Args:
            model_fn: the denoising network
            x_start: original clean data
            x_t: noisy data
            t: timestep of the noisy data (and the corresponding term of the bound
            to return)
        Returns:
            a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
            (specified by `t`), and `pred_x_start_logits` is logits of
            the denoised image.
        """
        true_logits = self.data_diffusor.q_posterior_logits(
            x_start, x_t, t, x_start_logits=False
        )
        model_logits, pred_x_start_logits = self.p_logits(
            x=x_t, t=t, model_output=output_logits
        )
        kl = self.categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        decoder_nll = -self.categorical_log_likelihood(x_start, model_logits)

        assert kl.shape == decoder_nll.shape == x_start.shape
        individual_atomic_loss = torch.where(t == 0, decoder_nll, kl)
        assert individual_atomic_loss.shape == x_start.shape
        total_loss = individual_atomic_loss.mean(
            dim=list(range(1, len(kl.shape)))
        ) / torch.log(torch.tensor(2.0))
        assert (
            total_loss.dim() == 0
        )  # we want to average everything to get a single scalar loss
        return total_loss, pred_x_start_logits

        # this was the old loss calculation logic
        # kl = kl.mean(dim=list(range(1, len(kl.shape)))) / torch.log(torch.tensor(2.0))
        # decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / torch.log(torch.tensor(2.0))

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        # assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)

        # curtis: this was the old return val
        # return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    # --------------------------------------
    # Log likelihood Loss
    # --------------------------------------

    def categorical_log_likelihood(self, x, logits):
        """Log likelihood of a discretized Gaussian specialized for image data.
        Assumes data `x` consists of integers [0, num_classes-1].
        Args:
            x: where to evaluate the distribution. shape = (bs, ...), dtype=int64
            logits: logits, shape = (bs, ..., num_classes)
        Returns:
            log likelihoods
        """
        log_probs = F.log_softmax(logits, dim=-1)
        x_onehot = F.one_hot(x, logits.shape[-1]).to(torch.get_default_dtype())
        return torch.sum(log_probs * x_onehot, dim=-1)

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate cross-entropy between x_start and predicted x_start.
        Args:
            x_start: original clean data
            pred_x_start_logits: predicted_logits
        Returns:
            ce: cross-entropy.
        """
        ce = -self.categorical_log_likelihood(x_start, pred_x_start_logits)
        assert ce.shape == x_start.shape
        ce = ce.mean(dim=list(range(1, len(ce.shape)))) / torch.log(torch.tensor(2.0))
        # assert ce.shape == (x_start.shape[0],)
        assert ce.dim() == 0  # we want just one number for the loss
        return ce

    # --------------------------------------
    # coordinate score matching loss
    # --------------------------------------

    def coord_score_matching_loss(self, coord_score, x_t):
        loss, _grad1, _grad2 = single_sliced_score_matching(coord_score, x_t)
        return loss

    # --------------------------------------
    # Perform the forward process
    # --------------------------------------

    def forward(self, *, batch, output):
        a_start = torch.argmax(batch["A0"], dim=-1)
        a_t = torch.argmax(batch["At"], dim=-1)
        x_t = batch["positions"]
        l_t = batch["Lt"]
        t = batch["t"]
        pred_A0_given_At = output["pred_A0_given_At"]
        coord_score = output["coord_score"]
        # print("target:", x_start, "input: ", x_t, "predicted:", torch.argmax(output_logits, dim=-1))
        """Training loss calculation."""
        # # Add noise to data
        # noise = torch.rand(x_start.shape + (self.num_atomic_states,), device=x_start.device)
        # t = torch.randint(0, self.num_timesteps, size=(x_start.shape[0],), device=x_start.device)
        # # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # # itself.
        # x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        vb_losses, pred_x_start_logits = self.vb_terms_bpd(
            x_start=a_start, x_t=a_t, t=t, output_logits=pred_A0_given_At
        )
        ce_losses = self.cross_entropy_x_start(
            x_start=a_start, pred_x_start_logits=pred_x_start_logits
        )

        score_matching_losses = self.coord_score_matching_loss(coord_score, x_t)

        # losses = vb_losses + self.hybrid_coeff * ce_losses + self.score_matching_coeff * score_matching_losses
        losses = score_matching_losses
        print("losses:", losses.item())
        # print("losses:", losses.item(), "target:", x_start, "input:", x_t, "position:", batch["positions"],"predicted:", torch.argmax(output_logits, dim=-1))

        assert losses.dim() == 0
        return losses
