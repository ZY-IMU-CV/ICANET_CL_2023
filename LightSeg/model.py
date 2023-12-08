from blocks import *

class RegSeg(nn.Module):
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False, change_num_classes=False):
        super().__init__() 
        body_name, decoder_name = name.split("_")
        if "LightSegE0" == body_name:
            self.encoder = LightSeg_Encoder_0()
        elif "LightSegE1"== body_name:
            self.encoder = LightSeg_Encoder_1()
        elif "LightSegE2"== body_name:
            self.encoder = LightSeg_Encoder_2()
        elif "LightSegE3"== body_name:
            self.encoder = LightSeg_Encoder_3()
        elif "LightSegE4"== body_name:
            self.encoder = LightSeg_Encoder_4()
        elif "LightSegE5"== body_name:
            self.encoder = LightSeg_Encoder_5()
        elif "LightSegE6"== body_name:
            self.encoder = LightSeg_Encoder_6()
        elif "LightSegE7"== body_name:
            self.encoder = LightSeg_Encoder_7()
        #elif "LightSegE8"== body_name:
            #self.encoder = LightSeg_Encoder_8()
        elif "LightSegE9"== body_name:
            self.encoder = LightSeg_Encoder_9()
        elif "LightSegE10"== body_name:
            self.encoder = LightSeg_Encoder_10()
        elif "LightSegE11"== body_name:
            self.encoder = LightSeg_Encoder_11()
        elif "LightSegE12"== body_name:
            self.encoder = LightSeg_Encoder_12()
        elif "LightSegE13"== body_name:
            self.encoder = LightSeg_Encoder_13()
        elif "LightSegE14"== body_name:
            self.encoder = LightSeg_Encoder_14()
            
        elif "LightSegHE0"== body_name:
            self.encoder = LightSeg_H_Encoder_0()
        elif "ResNet18" == body_name:
            self.encoder = ResNet_18()
        elif "LightSegR18E0" == body_name:
            self.encoder = LightSeg_R18_Encoder_0()
        else:
            raise NotImplementedError()
        if "LightSegD0" == decoder_name:
            self.decoder = LightSeg_Decoder0(num_classes, self.encoder.channels())
        elif "LightSegD1"==decoder_name:
            self.decoder = LightSeg_Decoder1(num_classes, self.encoder.channels())
        elif "LightSegD2"==decoder_name:
            self.decoder = LightSeg_Decoder2(num_classes, self.encoder.channels())
        elif "LightSegD3"==decoder_name:
            self.decoder = LightSeg_Decoder3(num_classes, self.encoder.channels())
        elif "LightSegD4"==decoder_name:
            self.decoder = LightSeg_Decoder4(num_classes, self.encoder.channels())
        elif "LightSegD5"==decoder_name:
            self.decoder = LightSeg_Decoder5(num_classes, self.encoder.channels())
        elif "LightSegD6"==decoder_name:
            self.decoder = LightSeg_Decoder6(num_classes, self.encoder.channels())
        elif "LightSegD7"==decoder_name:
            self.decoder = LightSeg_Decoder7(num_classes, self.encoder.channels())
        elif "LightSegD8"==decoder_name:
            self.decoder = LightSeg_Decoder8(num_classes, self.encoder.channels())
        else:
            raise NotImplementedError()
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            print(type(dic))
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            print(type(dic))
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x):
        input_shape=x.shape[-2:]
        x=self.encoder(x)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
