import math
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import sigmoid_beta_schedule
import networkx as nx
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, MarkovKernelApplication
from .generative import *
from .inference import DcpcGraphicalModel, asvi, mlp_amortizer

class BouncingMnistAsvi(ImportanceModel):
    def __init__(self, digit_side=28, hidden_dim=400, num_digits=3, T=10,
                 x_side=96, z_what_dim=10, z_where_dim=2):
        super().__init__()
        self._num_digits = num_digits

        self.encoders = pnn.PyroModule[nn.ModuleDict]({
            'mean_fields': pnn.PyroModule[nn.ModuleDict]({
                'z_what': pnn.PyroModule[nn.ModuleDict]({
                    'loc': mlp_amortizer(T * x_side ** 2, hidden_dim,
                                         num_digits * z_what_dim),
                    'scale': mlp_amortizer(T * x_side ** 2, hidden_dim,
                                           num_digits * z_what_dim),
                }),
                'z_where': pnn.PyroModule[nn.ModuleDict]({
                    'loc': mlp_amortizer(x_side ** 2, hidden_dim,
                                         num_digits * z_where_dim),
                    'scale': mlp_amortizer(x_side ** 2, hidden_dim,
                                           num_digits * z_where_dim),
                })
            }),
            'mixing_logits': pnn.PyroModule[nn.ModuleDict]({
                'z_what':  mlp_amortizer(T * x_side ** 2,
                                         num_digits * z_what_dim, 1),
                'z_where':  mlp_amortizer(x_side ** 2, num_digits * z_where_dim,
                                          1)
            }),
        })

        self.decoder = DigitsDecoder(digit_side, hidden_dim, x_side, z_what_dim)
        self.digit_features = DigitFeatures(z_what_dim)
        self.digit_positions = DigitPositions(z_where_dim)

    def generate(self, xs):
        B, T, _, _ = xs.shape
        pz_what = self.digit_features(K=self._num_digits, batch_shape=(B,))
        z_what = pyro.sample("z_what", pz_what)
        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            pz_where = self.digit_positions(z_where, K=self._num_digits,
                                            batch_shape=(B,))
            z_where = pyro.sample("z_where__%d" % t, pz_where)
            px = self.decoder(z_what, z_where)
            pyro.sample("X__%d" % t, px, obs=x)

    def guide(self, xs):
        B, T, _, _ = xs.shape

        data = xs.reshape(xs.shape[0], math.prod(xs.shape[1:]))
        with asvi(amortizer=self.encoders, data=data, event_shape=xs.shape[1:],
                  namer=lambda n: n.split('__')[0]):
            pz_what = self.digit_features(self._num_digits, batch_shape=(B,))
            pyro.sample("z_what", pz_what)

        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            x = x.reshape(x.shape[0], math.prod(x.shape[1:]))
            with asvi(amortizer=self.encoders, data=x, event_shape=x.shape[1:],
                      namer=lambda n: n.split('__')[0]):
                pz_where = self.digit_positions(z_where, K=self._num_digits,
                                                batch_shape=(B,))
                z_where = pyro.sample('z_where__%d' % t, pz_where)

class MnistDcpc(DcpcGraphicalModel):
    def __init__(self, x_dims, z_dims=[20, 128, 256]):
        super().__init__()
        self.prior = GaussianPrior(z_dims[0])
        self.decoder1 = ConditionalGaussian(z_dims[0], z_dims[1])
        self.decoder2 = ConditionalGaussian(z_dims[1], z_dims[2])
        self.likelihood = MlpBernoulliLikelihood(z_dims[2],
                                                 (x_dims[-1], x_dims[-1]))

        self.add_node("z1", [], MarkovKernelApplication("prior", (), {}))
        self.add_node("z2", ["z1"], MarkovKernelApplication("decoder1", (),
                                                            {}))
        self.add_node("z3", ["z2"], MarkovKernelApplication("decoder2", (),
                                                            {}))
        self.add_node("X", ["z3"], MarkovKernelApplication("likelihood", (),
                                                           {}))

    def conditioner(self, data):
        return {"X": data}

class BouncingMnistDcpc(DcpcGraphicalModel):
    def __init__(self, dims, digit_side=28, hidden_dim=400, num_digits=3,
                 z_what_dim=10, z_where_dim=2):
        super().__init__()
        self._num_digits = num_digits
        self._num_times = dims[0]

        self.decoder = DigitsDecoder(digit_side, hidden_dim, dims[-1],
                                     z_what_dim)
        self.digit_features = DigitFeatures(num_digits, z_what_dim)
        self.digit_positions = DigitPositions(num_digits=num_digits,
                                              z_where_dim=z_where_dim)

        self.add_node("z_what", [],
                      MarkovKernelApplication("digit_features", (), {}))
        for t in range(dims[0]):
            if t == 0:
                where_kernel = MarkovKernelApplication("digit_positions",
                                                       (None,), {})
                self.add_node("z_where__0", [], where_kernel)
            else:
                self.add_node(
                    "z_where__%d" % t, ["z_where__%d" % (t-1)],
                    MarkovKernelApplication("digit_positions", (), {})
                )
            self.add_node("X__%d" % t, ["z_what", "z_where__%d" % t],
                          MarkovKernelApplication("decoder", (), {}))

    def conditioner(self, xs):
        T = xs.shape[1]
        return {"X__%d" % t: xs[:, t] for t in range(T)}

class DiffusionDcpc(DcpcGraphicalModel):
    def __init__(self, dims, dim_mults=(1, 2, 4, 8), unet="Unet",
                 flash_attn=True, hidden_dim=64, T=100):
        super().__init__()
        self._channels = dims[0]
        self._num_times = T

        self.diffusion = DiffusionStep(sigmoid_beta_schedule(T),
                                       dim_mults=dim_mults,
                                       flash_attn=flash_attn,
                                       hidden_dim=hidden_dim,
                                       unet=unet, x_side=dims[-1])
        self.prior = DiffusionPrior(dims[0], dims[-1])

        self.add_node("X__%d" % T, [], MarkovKernelApplication("prior", (),
                                                               {}))
        for t in reversed(range(T)):
            step_kernel = MarkovKernelApplication("diffusion", (), {"t": t})
            self.add_node("X__%d" % t, ["X__%d" % (t+1)], step_kernel)

    def conditioner(self, data):
        return {"X__0": data}

class GeneratorDcpc(DcpcGraphicalModel):
    def __init__(self, dims, z_dim=40, heteroskedastic=True, hidden_dim=256,
                 discretize=True):
        super().__init__()
        self._channels = dims[0]
        self._prediction_subsample = 10000

        self.prior = GaussianPrior(z_dim, False)
        if heteroskedastic:
            self.likelihood = ConvolutionalDecoder(dims[0], z_dim, dims[-1],
                                                   discretize=discretize,
                                                   hidden_dim=hidden_dim)
        else:
            self.likelihood = FixedVarianceDecoder(dims[0], img_side=dims[-1],
                                                   discretize=discretize,
                                                   z_dim=z_dim)

        self.add_node("z", [], MarkovKernelApplication("prior", (), {}))
        self.add_node("X", ["z"], MarkovKernelApplication("likelihood", (),
                                                          {}))

        self.gmm = None

    def conditioner(self, data):
        return {"X": data}

    def predict(self, *args, B=1, P=1, z=None):
        from sklearn.mixture import GaussianMixture
        z = z.flatten(0, 1)
        idx = torch.randint(0, z.shape[0], (self._prediction_subsample,))
        if self.gmm is None:
            self.gmm = GaussianMixture(n_components=100).fit(z[idx])
        assignments = dist.Categorical(probs=torch.tensor(self.gmm.weights_))
        locs = torch.tensor(self.gmm.means_).to(dtype=torch.float)
        tril = torch.tril(torch.tensor(self.gmm.covariances_))
        tril = tril.to(dtype=torch.float)

        cs = assignments.sample((B,))
        zs = dist.MultivariateNormal(locs[cs], scale_tril=tril[cs])((P,))
        zs = zs.to(device=self.prior.loc.device)
        predictive = self.kernel("X")(zs.view(P, B, -1))
        return predictive.base_dist.loc

class ConvolutionalVae(ImportanceModel):
    def __init__(self, dims, discretize=True, heteroskedastic=True, z_dim=40,
                 hidden_dim=256):
        super().__init__()
        self._channels = dims[0]
        self._prediction_subsample = 10000

        if heteroskedastic:
            self.decoder = ConvolutionalDecoder(dims[0], z_dim, dims[-1],
                                                discretize=discretize,
                                                hidden_dim=hidden_dim)
        else:
            self.decoder = FixedVarianceDecoder(dims[0], img_side=dims[-1],
                                                discretize=discretize,
                                                z_dim=z_dim)
        self.encoder = ConvolutionalEncoder(self._channels, z_dim,
                                            hidden_dim=hidden_dim,
                                            img_side=dims[-1])
        self.prior = GaussianPrior(z_dim, False)

    def model(self, xs=None, **kwargs):
        B, _, _, _ = xs.shape
        with pyro.plate_stack("data", (B,)):
            z = pyro.sample("z", self.prior())
            return pyro.sample("X", self.decoder(z, obs=xs), obs=xs)

    def guide(self, xs: torch.Tensor, **kwargs):
        B, _, _, _ = xs.shape
        with pyro.plate_stack("data", (B,)):
            return pyro.sample("z", self.encoder(xs).to_event(1))

    def predict(self, *args, B=1, P=1, z=None, **kwargs):
        zs = z.to(device=self.prior.loc.device)
        model = pyro.condition(self.model, data={"X": None, "z": zs.view(P, B, -1)})
        trace = pyro.poutine.trace(model).get_trace(*args, B=B, P=P)
        likelihood = trace.nodes['X']["fn"]
        while hasattr(likelihood, "base_dist"):
            likelihood = likelihood.base_dist
        return likelihood.mean

class SequentialMemoryDcpc(DcpcGraphicalModel):
    def __init__(self, dims, z_dim=480, u_dim=0, nonlinearity=nn.Tanh):
        super().__init__()
        self._num_times, C, W, H = dims
        self._u_dim = u_dim
        self._x_shape = (C, W, H)

        self.emission = GaussianEmission(z_dim, C*W*H, u_dim=u_dim,
                                         nonlinearity=nonlinearity)
        if u_dim:
            self.policy = GaussianPrior(u_dim)
        self.prior = GaussianPrior(z_dim)
        self.transition = GaussianSsm(z_dim, u_dim=u_dim,
                                      nonlinearity=nonlinearity)

        self.add_node("z__0", [], MarkovKernelApplication("prior", (), {}))
        for t in range(self._num_times):
            transition_parents = ["z__%d" % t]
            emission_parents = ["z__%d" % (t+1)]
            if self._u_dim:
                self.add_node("u__%d" % (t+1), [],
                              MarkovKernelApplication("policy", (), {}))
                transition_parents.append("u__%d" % (t+1))
                emission_parents.append("u__%d" % (t+1))

            self.add_node("z__%d" % (t+1), transition_parents,
                          MarkovKernelApplication("transition", (), {}))
            self.add_node("X__%d" % (t+1), emission_parents,
                          MarkovKernelApplication("emission", (), {}))

    def conditioner(self, xs):
        T = xs.shape[1]
        assert self._num_times == T
        return {"X__%d" % (t+1): xs[:, t].view(-1, math.prod(self._x_shape))
                for t in range(T)}
