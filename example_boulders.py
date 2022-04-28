#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.acid import *
from src.utils import *
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


image_directory='./images/67p.IMG'
models_directory = './models/*.h5'


image_crop = Read_Preprocess_Image(image_directory,NORMALIZE=1, CONV_GS=1,INVERSE=0,EQUALIZE=0, CLAHE=1,                              RESIZE=1 , LIMITS=[1535,1791,0,256])

Objects_Master_list = model_inference(image_crop,models_directory,which_models='all')


# In[3]:


objects_unique = get_unique_iou(Objects_Master_list,iou_thres=0.5,detection_thres=0.20)

objects_unique_readable = readable_output(objects_unique)

objects_unique_readable__ = objects_unique_readable[(objects_unique_readable['detection_thres'] > 0.) &                                                    (objects_unique_readable['ellipticity'] < 3.0) &                                                     (objects_unique_readable['object_size_pixels'] < 0.01*(512**2))]

totalmask = np.sum(objects_unique_readable__['mask'])
totalmask[totalmask>0] = 1



# In[ ]:





# In[4]:



fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20, 40.)
plt.style.use('classic')
matplotlib.style='classic'
plt.subplot(1,2,1)
plt.imshow(image_crop ,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(image_crop ,cmap='gray')
totalmaskMasked = np.ma.masked_where(totalmask == 0, totalmask)

plt.imshow(totalmaskMasked,alpha=0.5,cmap='cool')
plt.show()        

