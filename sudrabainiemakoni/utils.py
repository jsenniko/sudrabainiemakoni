__author__ = 'Juris Seņņikovs'
import datetime
import exifread
import skimage.color
import numpy as np
def dateFromExif(filename: str) -> datetime.datetime:
    f = open(filename, 'rb')
    tags = exifread.process_file(f)
    x=tags['EXIF DateTimeOriginal']
    exifdate =datetime.datetime.strptime(x.values, '%Y:%m:%d %H:%M:%S')
    f.close()
    return exifdate
def getExifTags(filename):
    import exifread
    f = open(filename, 'rb')
    tags = exifread.process_file(f)
    f.close()
    return tags
# gamma adjust funkciju paņēmu no savas raustīgo time-lapsu korekcijas (sk piem 2020.g. komētu time-lapses)
def gamma_adjust(img_fixed, img_test):
    lthreshold=5
    l1=skimage.color.rgb2lab(img_fixed)[:,:,0]
    l1=l1[l1>lthreshold]
    l2=skimage.color.rgb2lab(img_test)[:,:,0]
    l2=l2[l2>lthreshold]
    #print(l1)
    #print(l2)
    L1=l1.mean()
    L2=l2.mean()
    gamma = np.log(L1/100)/np.log(L2/100)
    print('adjustgamma', gamma)
    imgtransformed=skimage.exposure.adjust_gamma(img_test,gamma=gamma)
    return imgtransformed

def writeWithText(imgarr, z, filename):
    from PIL import Image, ImageDraw, ImageFont
    pimg = Image.fromarray(np.uint8(imgarr))
    draw = ImageDraw.Draw(pimg)
    font = ImageFont.truetype("arial.ttf", 72)
    draw.text((10, 10),str(z)+' km',(128,255,128),font=font)
    pimg.save(filename)


def getAverageImages(imgarr):
    img_adjust=[imgarr[0]]
    for img in imgarr[1:]:
        img_adj=gamma_adjust(imgarr[0],img)
        img_adjust.append(img_adj)
    img_adjust=np.array(img_adjust)

    weights=np.repeat(1/len(imgarr), len(imgarr))
    img_mean=(img_adjust*weights[:,np.newaxis, np.newaxis,np.newaxis]).sum(axis=0)

    img_bw=[]
    for img in img_adjust:
        img_bw.append(skimage.color.rgb2gray(img))
    img_bw=np.array(img_bw)
    minthr=2
    img_max=img_bw.max(axis=0)
    img_adjust_m1=np.where(img_bw>minthr, img_bw, 1.0)
    img_min=img_adjust_m1.min(axis=0)
    xxx=np.zeros(shape=imgarr[0].shape)
    v=(img_max-img_min)#*1.5+0.4
    xxx[:,:,0]=v
    xxx[:,:,1]=v
    xxx[:,:,2]=v
    img_diff=np.array(xxx*255, dtype='uint8')
    img_tricolor=None
    if len(img_bw)==3:
        xxx=np.zeros(shape=imgarr[0].shape)
        xxx[:,:,0]=img_bw[0]
        xxx[:,:,1]=img_bw[1]
        xxx[:,:,2]=img_bw[2]
        img_tricolor=np.array(xxx*255, dtype='uint8')
    img_bicolor={}
    for i in range(len(img_bw)):
        for j in range(i+1,len(img_bw)):
            xxx=np.zeros(shape=imgarr[0].shape)
            v=((1-img_bw[i])+img_bw[j])/2.0
            xxx[:,:,0]=v
            xxx[:,:,1]=v
            xxx[:,:,2]=v
            img=np.array(xxx*255, dtype='uint8')
            img_bicolor[(i, j)]=img
    return img_mean, img_diff, img_tricolor, img_bicolor

def GetRayConvergence(p1, v1, p2, v2):
    matr=[[np.dot(v1, v1),-np.dot(v1, v2)],
      [np.dot(v2, v1),-np.dot(v2, v2)]]
    minv=np.linalg.inv(matr)
    (dist1, dist2)=np.dot(minv,np.array([np.dot(p2-p1,v1),np.dot(p2-p1,v2)]))
    #print(dist1,dist2)
    pclosest1 = (p1+dist1*v1)
    pclosest2 = (p2+dist2*v2)
    distance = np.linalg.norm(pclosest2-pclosest1)
    midpoint = 0.5*(pclosest1+pclosest2)
    return midpoint, distance, pclosest1, pclosest2, dist1, dist2
