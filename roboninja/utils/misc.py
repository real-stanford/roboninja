import collections
import os
import shutil
import socket

import dominate
import imageio
import numpy as np
import ray
from moviepy.editor import ImageSequenceClip


def get_vulkan_offset():
    hostname = socket.gethostname()
    if hostname == 'crv03':
        return 1
    else:
        return 0

def animate(imgs, filename='animation.mp4', _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=False)

def mkdir(path, clean=False):
    """Make directory.
    
    Args:
        path: path of the target directory
        clean: If there exist such directory, remove the original one or not
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def imretype(im, dtype):
    """Image retype.
    
    Args:
        im: original image. dtype support: float, float16, float32, float64, uint8, uint16
        dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    Returns:
        image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def meshwrite(filename, verts, colors, faces=None):
    """Save 3D mesh to a polygon .ply file.
    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n" %
            (verts[i, 0], verts[i, 1], verts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3]))

    ply_file.close()

def imwrite(path, obj):
    """Save Image.
    
    Args:
        path: path to save the image. Suffix support: png or jpg or gif
        image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', clean=True, html_file_name='index', group_ids=None):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: figure or text
                - figure: ndarray --> .png or [ndarrays] --> .gif
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        others: (optional) [dict]; other figures
            - name: string; name of the data, visualize using h2()
            - data: string or ndarray(image)
            - height: (optional) int; height of the image (default 256)
        title: (optional) string; title of the webpage (default 'visualization')
        clean: [bool] clean folder or not
        html_file_name: [str] html_file_name
        id_groups: list of (id_list, group_name)
    """
    mkdir(web_path, clean=clean)
    figure_path = os.path.join(web_path, 'figures')
    mkdir(figure_path, clean=clean)
    imwrite_ray = ray.remote(imwrite).options(num_cpus=1)
    obj_ids = list()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            obj_ids.append(imwrite_ray.remote(os.path.join(figure_path, key + '.png'), value))
        elif isinstance(value, list) and isinstance(value[0], np.ndarray):
            obj_ids.append(imwrite_ray.remote(os.path.join(figure_path, key + '.gif'), value))
    ray.get(obj_ids)
    
    group_ids = group_ids if group_ids is not None else [('', ids)]

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                    src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
                    
        for group_name, ids in group_ids:
            if group_name != '':
                dominate.tags.h2(group_name)
            with dominate.tags.table(border=1, style='table-layout: fixed;'):
                with dominate.tags.tr():
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                        dominate.tags.p('id')
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center'):
                            dominate.tags.p(col)
                for id in ids:
                    with dominate.tags.tr():
                        bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                            for part in id.split('_'):
                                dominate.tags.p(part)
                        for col in cols:
                            with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                                value = data[f'{id}_{col}']
                                if isinstance(value, str):
                                    dominate.tags.p(value)
                                elif isinstance(value, list) and isinstance(value[0], str):
                                    for v in value:
                                        dominate.tags.p(v)
                                elif isinstance(value, list) and isinstance(value[0], np.ndarray):
                                    dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.gif'.format(id, col)))
                                elif isinstance(value, np.ndarray):
                                    dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
                                else:
                                    raise NotImplementedError()
    
    with open(os.path.join(web_path, f'{html_file_name}.html'), 'w') as fp:
        fp.write(web.render())