# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ObjClsPredictor(nn.Module):
    def __init__(self, in_channels):
        super(ObjClsPredictor, self).__init__()
        num_classes = 38
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        return scores


class ROIBoxHeadACR1(torch.nn.Module):
    """
    Box Head class, for Action Recognition
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHeadACR1, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        # self.predictor = make_roi_box_predictor(
        #     cfg, self.feature_extractor.out_channels)
        self.obj_cls_predictor = ObjClsPredictor(self.feature_extractor.out_channels)
        self.obj_act_predictor = ObjClsPredictor(self.feature_extractor.out_channels)
        self.hum_act_predictor = ObjClsPredictor(self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # TODO-CSAMPAT: May be not required anymore
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        # if not self.training:
        #     result = self.post_processor((class_logits, box_regression), proposals)
        #     return x, result, {}

        # loss_classifier, loss_box_reg = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )
        # return (
        #     x,
        #     proposals,
        #     dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        # )
        conf_th = 0.7  # Only proposals with confidence more than this are considered
        better_proposals = list(zip(*[a.numpy() for a in torch.where(class_logits >= conf_th)]))
        obj_cls_logits, obj_act_logits, hum_act_logits = self.obj_cls_predictor(x), self.obj_act_predictor(x), \
                                                         self.hum_act_predictor(x)



def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHeadACR1(cfg, in_channels)
