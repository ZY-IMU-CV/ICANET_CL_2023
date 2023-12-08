import torch
import numpy as np
import val_data
import time
from thop import profile
from model import RegSeg
import yaml
import torch.cuda.amp as amp
class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes=exclude_classes

    def update(self, a, b):
        a=a.cpu()
        b=b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds=inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global=acc_global.item() * 100
        acc=(acc * 100).tolist()
        iu=(iu * 100).tolist()
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global=round(acc_global,2)
        IOU=[round(i,2) for i in iu]
        mIOU=sum(iu)/len(iu)
        mIOU=round(mIOU,2)
        reduced_iu=[iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
        mIOU_reduced=round(mIOU_reduced,2)
        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"

def evaluate(model, data_loader, device, confmat,mixed_precision,print_every,max_eval):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image) 
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i+1==max_eval:
                break
    return confmat
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ =="__main__":
    torch.cuda.set_device(0)
    #config_filename = "config/cityscapes_500epochs11.yaml"
    config_filename = "config/camvid_200epochs.yaml"
    with open(config_filename) as file:
        config = yaml.full_load(file)
    #config["dataset_dir"] = "../CLReg_prog/cityscapes_dataset/"  # If you put the dataset at another place, change this
    config["dataset_dir"] = "../CLReg_prog/camvid_dataset/"  # If you put the dataset at another place, change this
    
    config["class_uniform_pct"] = 2  # since we're only evalutaing, not training
    config["model_name"] = "LightSegHE0_LightSegD0"
    config["save_best_path"] = './pth_file/best_LightSeg_15_2.pth' 
    class_num = 19
    hist = np.zeros((class_num, class_num))
    #val_data = val_data.get_cityscapes("../CLReg_prog/cityscapes_dataset/",(1024,2048),(1024,2048),0.5,'val',4)
    val_data = val_data.get_camvid("../CLReg_prog/camvid_dataset/",720,720,0.5,'val',4)
    
    
    net = RegSeg(
            name=config["model_name"],
            num_classes=config["num_classes"],
        ).to('cuda')
    
    #net.load_state_dict(torch.load(config["save_best_path"],'cuda')['model'])
    print(net)
    #inpute = torch.randn(1,3,1024,2048)
    inpute = torch.randn(1,3,720,960)
    
    inpute = inpute.to('cuda')
    flops , params = profile(net, inputs = (inpute,))
    print(flops/1000000000)
    print(params/1000000)
    ##############FPS####################
    
    F=0
    with torch.no_grad(): 
        
        net1 = RegSeg(
            name=config["model_name"],
            num_classes=config["num_classes"],
        ).to('cuda')
        #net1.load_state_dict(torch.load(config["save_best_path"], 'cuda')['model'])
        
        net1.eval()
        #comfort = ConfusionMatrix(19, [])
        #comfort = evaluate(net1, val_data, 'cuda', comfort, True, 5, 500)
        #print(comfort)
        for i,(image, target) in enumerate(val_data):
            image = image.to('cuda')
            print(image.shape)
            torch.cuda.synchronize()
            start_time = time.time()
            output = net1(image)
            torch.cuda.synchronize()
            end_tiem = time.time()
            time_sum = end_tiem - start_time
            print(time_sum)
            print(i)
            F+=time_sum
        print(F)
        #FPS = F/500
        FPS = F/140
    print(1/FPS)
    print(FPS)
    


