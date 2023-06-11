from PIL import Image
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import torch as th
import json
import sys,os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import tqdm
import clip
import cv2
th.set_grad_enabled(False)

def init(args,rank=None):
    global device,clip_model,preprocess,pipe
    global config_ldm,model_ldm,ldm_sampler, ldm_opt
    if rank is None:
        rank=th.multiprocessing.current_process()._identity[0]-1
    print("init process GPU:",rank)
    device = th.device('cuda:%s'%rank)
    th.cuda.set_device(device)
    th.cuda.empty_cache()
    clip_model, preprocess = clip.load("ViT-L/14", device=rank)
    if args.model=='stable-diffusion':
        sys.path.append(args.stable_diffusion_dir)
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.models.diffusion.plms import PLMSSampler
        ldm_opt=edict(dict(scale=5.0,ddim_steps=200,ddim_eta=0,H=512,W=512))
        config_ldm = OmegaConf.load(os.path.join(args.stable_diffusion_dir,"configs/stable-diffusion/v1-inference.yaml"))
        model_ldm = load_model_from_config(config_ldm, os.path.join(args.stable_diffusion_dir,"models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")).to(device)
        ldm_sampler = PLMSSampler(model_ldm)
    if args.model=='diffusers':
        model_id= "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16,output_type='latent')   
        pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet.enable_xformers_memory_efficient_attention()
        pipe.to(device)



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def sample_ldm(opt,prompt='',batch_size=1):
    with model_ldm.ema_scope():
        uc = None
        if opt.scale != 1.0:
            uc = model_ldm.get_learned_conditioning(batch_size * [""])
        c = model_ldm.get_learned_conditioning(batch_size * [prompt])
        shape = [4, opt.H//8, opt.W//8]
        samples_ddim, _ = ldm_sampler.sample(S=opt.ddim_steps,
                                         conditioning=c,
                                         batch_size=batch_size,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=opt.scale,
                                         unconditional_conditioning=uc,
                                         eta=opt.ddim_eta)

        x_samples_ddim = model_ldm.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)
    return x_samples_ddim


def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    pil_image=Image.fromarray(reshaped.numpy())
    #display(pil_image)
    return pil_image



def save_images(batch,path,cls,offset):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(0, 2, 3, 1).numpy()
    os.makedirs(os.path.join(path,cls),exist_ok=True)
    for i,j in enumerate(reshaped):
        pil_image=Image.fromarray(j)
        pil_image.save(os.path.join(path,cls,f"{offset+i:04d}.png"))

def gen_image(cls_info,method='latent',batch_size=10,total=100,output_path='/mnt/home/syn4det/LVIS_gen_FG_2'):
    text='a photo of a single {}'.format(' '.join(cls_info['name'].split('_')))
    clips=[]
    offset=0
    while 1:
        if method=='diffusers':
            latents=pipe(text, num_inference_steps=50, guidance_scale=7.5,num_images_per_prompt=batch_size).images
            latents = 1 / pipe.vae.config.scaling_factor * latents
            im = pipe.vae.decode(latents).sample
        else:
            im=sample_ldm(opt=ldm_opt,prompt=text,batch_size=batch_size)

        text_feature = clip.tokenize(text).cuda()
        _, logits_per_text = clip_model(torch.stack([preprocess(show_images(i.unsqueeze(0))) for i in im],0).cuda(), text_feature)
        logits_per_text=logits_per_text.view(-1).cpu().tolist()
        clips+=logits_per_text    
        save_images(im,output_path,cls_info['name'],offset)
        offset+=batch_size
        if offset>=total:
            return clips
        
    

from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import argparse
import os
d1=defaultdict(list)
d2=defaultdict(list)
if __name__=="__main__":
    _ = clip.load("ViT-L/14")
    del _
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diffusers')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='LVIS_gen_FG')
    parser.add_argument('--category_file', type=str, default='/mnt/data/LVIS/lvis_v1_train.json')
    parser.add_argument('--stable_diffusion_dir', type=str, default=os.path.join(os.getcwd(),'stable-diffusion'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.category_file) as f:
        data=json.load(f)
    target_class=[]
    for i in data['categories']:
        target_class.append(i)
    PATH=args.output_dir
    os.makedirs(PATH,exist_ok=True)
    if args.resume:
        old=os.listdir(PATH)
        cls_names=[j['name'] for j in target_class]
        for i in cls_names:
            try:
                _=cv2.imread(os.path.join(PATH,f"{i}_{args.samples-1}.png"))
                cls_names.remove(i)
            except:
                pass
        target_class=[i for i in target_class if i['name'] in cls_names]
    mp.set_start_method('spawn',force=True)
    pool = mp.Pool(processes=th.cuda.device_count(),initializer=init,initargs=(args,))
    results = pool.starmap(gen_image, [(i,args.model,args.batchsize,args.samples,PATH) for i in target_class],1)
    for i,j in zip(target_class,results):
        i['clip_scores']=j
    with open(os.path.join(PATH,"results.json"),'w') as f:
        json.dump(target_class,f)



