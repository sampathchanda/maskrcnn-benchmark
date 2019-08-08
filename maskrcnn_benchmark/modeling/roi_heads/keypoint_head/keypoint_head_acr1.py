import torch

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor
from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator


class ROIKeypointHeadACR1(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIKeypointHeadACR1, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # if self.training:
        #     with torch.no_grad():
        #         proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals)
        kp_logits = self.predictor(x)

        # if not self.training:
        result = self.post_processor(kp_logits, proposals)
        k_anchor = [torch.einsum("ij,i->j",
                                 [res.__dict__['bbox'], torch.nn.Softmax()(res.__dict__['extra_fields']['scores'])])
                    for res in result]

        return x, result, k_anchor

        # return x, result, {}
        # loss_kp = self.loss_evaluator(proposals, kp_logits)
        # return x, proposals, dict(loss_kp=loss_kp)


def build_roi_keypoint_head(cfg, in_channels):
    return ROIKeypointHeadACR1(cfg, in_channels)
