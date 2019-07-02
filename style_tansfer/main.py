#!/usr/bin/env python
# -*- coding: utf-8 -*-

import style_transfer

vgg_model_path = './style_transfer_data/vgg16.npy'
content_img_path = './style_transfer_data/gugong.jpg'
style_img_path = './style_transfer_data/xingkong.jpeg'
output_dir = './run_style_transfer'

model = style_transfer.style_transfer_v1(content_img_path, style_img_path, 
             vgg_model_path, output_dir=output_dir)
model.style_transfer_graph()
model.train()   