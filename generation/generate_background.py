from PIL import Image
import torch as th
import json
import sys,os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import tqdm
import random
import argparse
th.set_grad_enabled(False)

def init(args,rank=None):
    global pipe
    if rank is None:
        rank=th.multiprocessing.current_process()._identity[0]-1
    print("init process GPU:",rank)
    device = th.device('cuda:%s'%rank)
    th.cuda.set_device(device)
    th.cuda.empty_cache()
    model_id= "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16)   
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.enable_xformers_memory_efficient_attention()
    pipe.to(device)


def gen_image(id,prompt,batch_size=10,output_path='/mnt/home/syn4det/LVIS_gen_BG'):
    images=pipe(prompt, num_inference_steps=50, guidance_scale=7.5,num_images_per_prompt=batch_size).images
    for image,index in zip(images,range(id*batch_size,(id+1)*batch_size)):
        image.save(os.path.join(output_path,'{:06d}.png'.format(index)))
        
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--output', type=str, default='LVIS_gen_BG')
    args = parser.parse_args()
    PATH=os.path.join('/mnt/home/syn4det/',args.output)
    os.makedirs(PATH,exist_ok=True)
    prompt_pool=['a photo of city scene.','a photo of street.','a photo in the room.', 'a photo of nature', 'a photo of village', 'a photo of wild land' ]
    n_load=args.samples//args.batchsize+1
    prompts=random.choices(prompt_pool,k=n_load)
    mp.set_start_method('spawn',force=True)
    pool = mp.Pool(processes=th.cuda.device_count(),initializer=init,initargs=(args,))
    results = pool.starmap(gen_image, [(i,j,args.batchsize,PATH) for i,j in enumerate(prompts)])


