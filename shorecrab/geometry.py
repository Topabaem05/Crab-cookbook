"""Resolution-independent native tiling; deployment budgets are separate."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import numpy as np
from PIL import Image, ImageOps
import torch

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)
FILL=tuple(round(c*255) for c in MEAN)

class ResourceExhausted(RuntimeError):
    """The whole request is rejected, never silently partially evaluated."""

@dataclass(frozen=True)
class Tiling:
    canvas:int=512
    core:int=480
    halo:int=16
    locals:str='always'  # 'beyond_canvas': images that fit the canvas are read from the overview alone (LT1)
    def __post_init__(self):
        if any(type(x) is not int for x in (self.canvas,self.core,self.halo)):
            raise ValueError('tiling fields must be integers')
        if self.locals not in {'always','beyond_canvas'}:raise ValueError('tiling.locals must be always or beyond_canvas')
        if self.canvas<32 or self.canvas%32 or self.core<1 or self.halo<0 or self.core+2*self.halo>self.canvas:
            raise ValueError('canvas must be a positive multiple of 32; core+2*halo<=canvas')

@dataclass(frozen=True)
class Tile:
    index:int
    core:tuple[int,int,int,int]
    crop:tuple[int,int,int,int]

class TilePlan:
    def __init__(self,width:int,height:int,tiling:Tiling):
        if any(type(v) is not int or v<=0 for v in (width,height)):
            raise ValueError('width and height must be positive integers')
        self.width,self.height,self.tiling=width,height,tiling
        self.nx=(width+tiling.core-1)//tiling.core
        self.ny=(height+tiling.core-1)//tiling.core
        if tiling.locals=='beyond_canvas' and width<=tiling.canvas and height<=tiling.canvas:self.nx=self.ny=0
        self.count=self.nx*self.ny
    def __iter__(self)->Iterator[Tile]:
        w,h,t=self.width,self.height,self.tiling
        for j in range(self.ny):
            y0,y1=j*h//self.ny,(j+1)*h//self.ny
            for i in range(self.nx):
                x0,x1=i*w//self.nx,(i+1)*w//self.nx
                yield Tile(j*self.nx+i,(x0,y0,x1,y1),(max(0,x0-t.halo),max(0,y0-t.halo),min(w,x1+t.halo),min(h,y1+t.halo)))

@dataclass(frozen=True)
class Admission:
    max_file_bytes:int=33554432
    max_pixels:int=64000000
    max_tiles:int=256
    max_visual_tokens:int=16384
    def __post_init__(self):
        if any(type(v) is not int or v<=0 for v in vars(self).values()):
            raise ValueError('deployment budgets must be finite positive integers')
    def check(self,plan:TilePlan,global_queries:int=64,local_queries:int=32):
        if plan.width*plan.height>self.max_pixels: raise ResourceExhausted('decoded pixel budget exceeded')
        if plan.count>self.max_tiles: raise ResourceExhausted('tile work budget exceeded')
        if global_queries+local_queries*plan.count>self.max_visual_tokens:
            raise ResourceExhausted('compact memory budget exceeded')

def open_image(path:Path,tiling:Tiling,budget:Admission,global_queries=64,local_queries=32)->Image.Image:
    path=Path(path)
    if path.stat().st_size>budget.max_file_bytes: raise ResourceExhausted('encoded byte budget exceeded')
    with Image.open(path) as im:
        # Header admission before decoding; rotation does not change pixel/tile count.
        budget.check(TilePlan(*im.size,tiling),global_queries,local_queries)
        if getattr(im,'n_frames',1)!=1: raise ValueError('single raster image required, not animated input')
        im=ImageOps.exif_transpose(im)
        if 'A' in im.getbands() or 'transparency' in im.info:
            rgba=im.convert('RGBA'); bg=Image.new('RGBA',rgba.size,FILL+(255,)); bg.alpha_composite(rgba); im=bg.convert('RGB')
        else: im=im.convert('RGB')
        return im.copy()

@dataclass
class View:
    pixels:torch.Tensor       # [3,S,S], normalized CPU tensor
    owned:torch.Tensor        # [1,S,S], owned pixels (halo excluded for local views)
    kind:str                 # overview or local
    index:int
    source_size:tuple[int,int]
    origin:tuple[float,float]
    source_per_pixel:tuple[float,float]
    valid_size:tuple[int,int]
    source_box_fraction:tuple[float,float]

def _pixels(canvas:Image.Image)->torch.Tensor:
    x=torch.from_numpy(np.array(canvas,dtype=np.float32,copy=True)).permute(2,0,1)/255.
    return (x-torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None]

def iter_views(image:Image.Image,tiling:Tiling)->Iterator[View]:
    image=image.convert('RGB'); w,h=image.size; s=tiling.canvas
    scale=min(1.,s/w,s/h); rw=max(1,min(s,round(w*scale))); rh=max(1,min(s,round(h*scale)))
    thumb=image.resize((rw,rh),Image.Resampling.BICUBIC) if (rw,rh)!=(w,h) else image
    canvas=Image.new('RGB',(s,s),FILL); canvas.paste(thumb,(0,0))
    owner=torch.zeros(1,s,s);owner[:,:rh,:rw]=1
    yield View(_pixels(canvas),owner,'overview',-1,(w,h),(0.,0.),(w/rw,h/rh),(rw,rh),(1.,1.))
    for t in TilePlan(w,h,tiling):
        cx,cy,x1,y1=t.crop; cw,ch=x1-cx,y1-cy
        canvas=Image.new('RGB',(s,s),FILL);canvas.paste(image.crop(t.crop),(0,0))
        owner=torch.zeros(1,s,s)
        ox0,oy0,ox1,oy1=t.core;owner[:,oy0-cy:oy1-cy,ox0-cx:ox1-cx]=1
        yield View(_pixels(canvas),owner,'local',t.index,(w,h),(float(cx),float(cy)),(1.,1.),(cw,ch),(cw/w,ch/h))

def feature_geometry(view:View,fh:int,fw:int)->torch.Tensor:
    """Fixed source positions. Geometric support is NOT a CNN influence map."""
    s=view.pixels.shape[-1]; vw,vh=view.valid_size; w,h=view.source_size
    xs=(torch.arange(fw,dtype=torch.float32)+.5)*(s/fw)
    ys=(torch.arange(fh,dtype=torch.float32)+.5)*(s/fh)
    # Centers clipped to actual valid pixel extent; padding keys are later masked.
    xs=xs.clamp(0,max(vw-.5,0));ys=ys.clamp(0,max(vh-.5,0))
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    xx=(xx*view.source_per_pixel[0]+view.origin[0])/w
    yy=(yy*view.source_per_pixel[1]+view.origin[1])/h
    return torch.stack((xx,yy,torch.full_like(xx,view.source_box_fraction[0]),torch.full_like(yy,view.source_box_fraction[1])),-1).reshape(-1,4)

def geometry_embedding(geometry:torch.Tensor,dim:int)->torch.Tensor:
    if dim%8: raise ValueError('dim must be divisible by 8 for 4-scalar sinusoidal geometry')
    count=dim//8
    frequencies=(2*math.pi)*torch.pow(2.,torch.arange(count,device=geometry.device,dtype=torch.float32)/8)
    phase=geometry.float()[...,None]*frequencies
    return torch.cat((phase.sin(),phase.cos()),-1).reshape(*geometry.shape[:-1],dim)
