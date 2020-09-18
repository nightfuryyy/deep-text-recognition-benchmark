"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.gcn import GraphConvolution


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        map_backbon_to_len_sequence = {'VGG' : 24, 'RCNN' : 26, 'ResNet' : 26}
        self.len_sequence = map_backbon_to_len_sequence[opt.FeatureExtraction]
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.GraphConvolution_output = opt.output_channel_GCN
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        # if opt.SequenceModeling == 'BiLSTM':
        #     self.SequenceModeling = nn.Sequential(
        #         BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
        #         BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        #     self.SequenceModeling_output = opt.hidden_size
        # else :
        #     if opt.SequenceModeling == 'GCN-BiLSTM':
        #         self.SequenceModeling = nn.Sequential(
        #         GraphConvolution(opt.batch_size, self.len_sequence, self.FeatureExtraction_output, self.GraphConvolution_output, bias = True, scale_factor = 0),
        #         BidirectionalLSTM(self.GraphConvolution_output, opt.hidden_size, opt.hidden_size),
        #         BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        #         self.SequenceModeling_output = opt.hidden_size
        #     else:
        #         print('No SequenceModeling module specified')
        #         self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        # if opt.guide_training :
        self.SequenceModeling_CTC =  nn.Sequential(
            GraphConvolution(opt.batch_size, self.len_sequence, self.FeatureExtraction_output, self.GraphConvolution_output, bias = False, scale_factor = 0),
            BidirectionalLSTM(self.GraphConvolution_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
        self.SequenceModeling_Attn =  nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
        self.CTC = nn.Linear(opt.hidden_size, opt.num_class_ctc)
        self.Attention = Attention(opt.hidden_size, opt.hidden_size, opt.num_class_attn)


    def inference(self, input, text) :
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        contextual_feature_from_major_path = self.SequenceModeling_CTC(visual_feature)
        prediction_from_major_path = self.CTC(contextual_feature_from_major_path.contiguous())
        return prediction_from_major_path


    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)



        # """ Sequence modeling stage """
        # if self.stages['Seq'] == 'BiLSTM':
        #     contextual_feature = self.SequenceModeling(visual_feature)
        # elif self.stages['Seq'] == 'GCN-BiLSTM':
        #     contextual_feature = self.SequenceModeling(visual_feature)
        # else :
        #     contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        # if opt.guide_training :
        contextual_feature_from_major_path = self.SequenceModeling_CTC(visual_feature)
        prediction_from_major_path = self.CTC(contextual_feature_from_major_path.contiguous())
        contextual_feature_from_guide_path = self.SequenceModeling_Attn(visual_feature)
        prediction_from_guide_path = self.Attention(contextual_feature_from_guide_path.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        return prediction_from_major_path, prediction_from_guide_path
        """ Prediction stage """
        # if self.stages['Pred'] == 'CTC':
        #     prediction = self.Prediction(contextual_feature.contiguous())
        # else:
        #     prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        # return prediction

# if __name__ == '__main__':

