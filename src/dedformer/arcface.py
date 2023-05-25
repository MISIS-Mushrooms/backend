import math

import numpy as np
import scipy
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import WeightRegularizerMixin, BaseMetricLossFunction
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class ArcFace(WeightRegularizerMixin, BaseMetricLossFunction):

    def __init__(self, margin=28.6, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.scale = scale
        self.add_to_recordable_attributes(
            list_of_names=["num_classes", "margin", "scale"], is_stat=False
        )
        self.add_to_recordable_attributes(name="avg_angle", is_stat=True)
        self.init_margin()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def init_margin(self):
        self.margin = np.radians(self.margin)

    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        for attr in ["n_range", "margin_choose_n", "cos_powers", "alternating"]:
            setattr(self, attr, c_f.to_device(getattr(self, attr), cosine))
        cos_powered = cosine**self.cos_powers
        sin_powered = (1 - cosine**2) ** self.n_range
        terms = (
            self.alternating * self.margin_choose_n * cos_powered * sin_powered
        )  # Equation 7 in the paper
        return torch.sum(terms, dim=1)

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        if self.collect_stats:
            with torch.no_grad():
                self.avg_angle = np.degrees(torch.mean(angles).item())
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)

        # Compute cos of (theta + margin) and cos of theta
        cos_theta_plus_margin = torch.cos(angles + self.margin)
        cos_theta = torch.cos(angles)

        # Keep the cost function monotonically decreasing
        unscaled_logits = torch.where(
            angles <= np.deg2rad(180) - self.margin,
            cos_theta_plus_margin,
            cos_theta - self.margin * np.sin(self.margin),
        )

        return unscaled_logits

    def scale_logits(self, logits, embeddings):
        return logits * self.scale

    def cast_types(self, dtype, device):
        pass

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)
        unweighted_loss = self.cross_entropy(logits, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict

    def get_default_distance(self):
        return CosineSimilarity()

    def get_logits(self, embeddings):
        logits = self.get_cosine(embeddings)
        return self.scale_logits(logits, embeddings)