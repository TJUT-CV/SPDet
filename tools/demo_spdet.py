import argparse
import json
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger
import glob
from torchvision.ops import nms
from yolox.data.data_augment import preproc,preproc_new
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
import time
import numpy as np
from yolox.utils.bbox import xyxy2xywh,calc_IoU,Giou,xywh2xyxy,calc_IoU2,xyxy2cxywh
from cython_bbox import bbox_overlaps as bbox_ious
import warnings
warnings.filterwarnings('ignore')

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def ious_dis(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )
    return ious


def letterbox(img, height=800, width=1440, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def non_max_suppression(prediction, conf_thres=0.1, nms_thres=0.45, method='standard'):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        # pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'fast':
            nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        else:
            raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))
    return output
def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords

class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like
        detection result : Bounding box & confidence.
    feature : array_like
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, outputs, features):
        self.outputs = np.asarray(outputs, dtype=np.float)
        # self.confidence = float(confidence)
        self.features = np.asarray(features, dtype=np.float)
class LoadPandaImages:  # for inference
    def __init__(self, path,args, img_size=(800, 1440),predictor=None, overlap=0.2, scales=(0.2, 0.6)): #(1440, 800) (0.2,0.4)
        self.args = args
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.predictor=predictor
        self.height = img_size[0]
        self.width = img_size[1]
        self.seq_name=path.split('/')[-1] if len(path.split('/')[-1])>0 else path.split('/')[-2]
        self.count = 0
        self.overlap = overlap
        self.scales = scales
        self.x_step = round(self.width * (1 - self.overlap))
        self.y_step = round(self.height * (1 - self.overlap))
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.get_first_dets()

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self
    def set_last_dets(self,outputs):
        self.last_dets=outputs
    def get_first_dets(self):
        img_path=self.files[0]
        src_giga_img = cv2.imread(img_path)
        self.orgin_shape=src_giga_img.shape[:2]
        sub_img_list = self._sliding_window_orgin(src_giga_img)
        clone_flag = False
        for img_tuple in sub_img_list:
            tic1 = time.time()
            img02, img, loc_info = img_tuple
            im_blob = img.transpose(1, 2, 0)
            timer1 = Timer()
            outputs, img_info = self.predictor.inference(im_blob, timer1)
            timer1.toc()
            toc1=time.time()
            if outputs[0] is not None:
                disp = torch.zeros(outputs[0].size(), device='cuda:0')
                disp[:, 0], disp[:, 1] = loc_info[1:3]
                disp[:, 2], disp[:, 3] = loc_info[1:3]
                outputs[0][:, :4] = outputs[0][:, :4] / img_info['ratio']
                pred_sub = torch.add(outputs[0], disp)
                pred_sub[:, :4] = pred_sub[:, :4] / loc_info[0]
                if not clone_flag:
                    clone_flag = True
                    pred = pred_sub.clone()
                else:
                    pred = torch.cat((pred, pred_sub), 0)
            toc2=time.time()
        toc = time.time()
        if len(pred) > 0:
            tic = time.time()
            dets = non_max_suppression(pred.unsqueeze(0))[0].cpu()
            toc = time.time()

        outputs = dets.clone()
        temp_outputs=[]
        if outputs is not None:
            temp_outputs = outputs.numpy()
        self.first_dets=temp_outputs
        self.last_dets=self.first_dets

    def _sliding_window(self, src_giga_img):
        sub_img_list = []
        raw_height, raw_width = src_giga_img.shape[:2]

        scale=1
        temp_all_count = 0
        for scale in self.scales:#multi-scale;
            n = 0
            src_img = cv2.resize(src_giga_img, (int(raw_width * scale), int(raw_height * scale)))
            src_height, src_width = src_img.shape[:2]
            # sub image generate
            ini_y = 0
            while ini_y < src_height:
                ini_x = 0
                while ini_x < src_width:
                    img0 = src_img[ini_y:ini_y + self.height, ini_x:ini_x + self.width]
                    img, radio, _, _ = letterbox(img0, height=self.height, width=self.width) ####################
                    loc_info = (scale, ini_x, ini_y, radio, 0, 0)
                    temp_all_count+=1
                    # Normalize RGB
                    img = img.transpose(2, 0, 1)#BGR
                    img = np.ascontiguousarray(img, dtype=np.float32)

                    sub_img_list.append((img0, img, loc_info))
                    n += 1
                    if ini_x == src_width - self.width:
                        break
                    ini_x += self.x_step
                    if ini_x + self.width > src_width:
                        ini_x = src_width - self.width
                if ini_y == src_height - self.height:
                    break
                ini_y += self.y_step
                if ini_y + self.height > src_height:
                    ini_y = src_height - self.height
        return sub_img_list
    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]
        print('processing img on {}'.format(img_path))
        # Read image
        #tic = time.time()
        timer1 = Timer()
        timer1.tic()
        tic=time.time()
        src_giga_img = cv2.imread(img_path)  # BGR

        timer1.toc()
        print('1    JPEG decoding cost time: {} s'.format(timer1.average_time))
        # assert src_giga_img is not None, 'Failed to load ' + img_path
        timer2 = Timer()
        timer2.tic()
        sub_img_list = self._sliding_window_ac(src_giga_img) ############ adaptation cat
        timer2.toc()
        print('2    sliding window cutting cost time: {} s'.format(timer2.average_time))  # '  window number: ', len(sub_img_list)
        return img_path, src_giga_img, sub_img_list

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]
        # Read image
        src_giga_img = cv2.imread(img_path)  # BGR
        assert src_giga_img is not None, 'Failed to load ' + img_path

        sub_img_list = self._sliding_window(src_giga_img)

        return img_path, src_giga_img, sub_img_list

    def __len__(self):
        return self.nF  # number of files
    def get_xy(self,begin_xy, dets_background, num, txt_dets2,scale,temp_min_ge):
        img_w = dets_background.shape[1]
        img_h = dets_background.shape[0]

        img_w_ge = round(self.width * scale)
        img_h_ge = round(self.height * scale)
        out_i = 1
        temp_label = False
        temp_bbox = []
        txt_dets2 = np.array(txt_dets2)
        txt_dets2*=scale
        for i in range(1, num + 1 - begin_xy[0]):
            temp_x1 = int(begin_xy[0] * img_w_ge)
            temp_y1 = int(begin_xy[1] * img_h_ge)
            temp_x2 = int((begin_xy[0] + i) * img_w_ge)
            temp_y2 = int((begin_xy[1] + i) * img_h_ge)
            # if sum(sum((dets_background[temp_y1:temp_y2,temp_x1:temp_x2])))>0:

            if sum(sum((dets_background[temp_y1:temp_y2, temp_x1:temp_x2]))) > 0:
                break
            out_i += 1

        out_xy = [begin_xy[0] + out_i - 1, begin_xy[1] + out_i - 1]
        if out_xy == begin_xy:
            out_xy = [out_xy[0] + 1, out_xy[1] + 1]
        scale_flag = False
        scale_flag = True
        out_scale = 0
        if scale_flag:
            b1 = begin_xy
            if temp_min_ge > 0:
                out_xy = [out_xy[0] + temp_min_ge - 1, out_xy[1] + temp_min_ge - 1]
            b2 = out_xy
            b1 = [b1[0] * img_w_ge, b1[1] * img_h_ge]
            b2 = [b2[0] * img_w_ge, b2[1] * img_h_ge]
            temp_b = [[b1[0], b1[1], b2[0], b2[1]]]

            ious = ious_dis(temp_b, txt_dets2)

            temp_flag = ious > 0
            temp_flag = temp_flag[0]
            if temp_flag.any():
                temp_label = True
                mout = txt_dets2[temp_flag]
                mout_h = mout[:, 3] - mout[:, 1]
                scale_num = 3
                if len(mout_h) > 10:
                    mout_h = list(mout_h)
                    mout_h.remove(max(mout_h))
                    mout_h.remove(max(mout_h))
                temp_mean = (np.mean(mout_h) + np.max(mout_h)) / 2
                if temp_mean > img_h_ge / scale_num:
                    out_scale = int(temp_mean / (img_h_ge / scale_num))
                    if out_scale > temp_min_ge:
                        temp_min_ge = out_scale

        out_xy = [out_xy[0] + out_scale, out_xy[1] + out_scale]
        temp_bbox_new = []
        if temp_label:
            b1 = begin_xy
            b2 = out_xy
            b1 = [b1[0] * img_w_ge, b1[1] * img_h_ge]
            b2 = [b2[0] * img_w_ge, b2[1] * img_h_ge]
            temp_b = [[b1[0], b1[1], b2[0], b2[1]]]
            temp_iou = ious_dis(temp_b, txt_dets2)
            temp_iou_flag = temp_iou > 0
            temp_iou_flag = temp_iou_flag[0]
            mout = txt_dets2[temp_iou_flag]
            temp_bbox = mout
            for ti in range(len(temp_bbox)):
                temp_iou_bbox = temp_bbox[ti]
                temp_iou_box_2 = calc_IoU2(temp_iou_bbox, temp_b[0])
                if temp_iou_box_2 > 0.2:
                    temp_iou_bbox = xyxy2xywh(temp_iou_bbox)
                    temp_bbox_new.append(temp_iou_bbox)
        temp_x1 = int(begin_xy[0] * img_w_ge)
        temp_y1 = int(begin_xy[1] * img_h_ge)
        temp_x2 = int((out_xy[0]) * img_w_ge)
        temp_y2 = int((out_xy[1]) * img_h_ge)
        if len(temp_bbox_new) > 0:
            temp_label = True
        dets_background[temp_y1:temp_y2, temp_x1:temp_x2] = 0.5
        cv2.rectangle(dets_background, (int(temp_x1 + 5), int(temp_y1 + 5)),
                      (int(temp_x2 - 5), int(temp_y2 - 5)), (200, 100, 200), 4)
        return out_xy, dets_background, temp_label, temp_bbox_new, temp_min_ge
    

    def get_h(self,dets_background, num,scale):
        img_h_ge = round(self.height*scale)
        out_h = 0
        for i in range(1, num + 1):
            temp_h = int(i * img_h_ge)
            if 0 in dets_background[0:temp_h, :]:
                break
            out_h = i
        return out_h


    def _sliding_window_ac(self, src_giga_img):
        sub_img_list = []
        scale=0.6
        n = 0
        show_time=1
        all_temp_count = 0
        scale = 0.2  ## 0.62s 1.61fps
        scale = 1  ## 0.277s 0.44fps
        # if True:  # multi-scale;

        src_height, src_width = src_giga_img.shape[:2]
        # src_height, src_width = src_img.shape[:2]
        temp_scales=[0.6,1]
        temp_scales=[0.6]
        temp_scales=[0.2]
        temp_scales=[1]


        # for self_scale in temp_scales:  # multi-scale;
        if True:
            self_scale=1
            n = 0
            if self_scale!=1:
                src_img = cv2.resize(src_giga_img, (int(src_width * self_scale), int(src_height * self_scale)))
            else:
                src_img = src_giga_img
            all_xy = []
            all_xy2 = []
            end_xy = [0, 0]
            count_w = 0
            showwidth = 1920
            # showwidth = src_img.shape[1]
            temp_min_ge = 0
            all_label = []
            all_bbox = []
            scale = showwidth / src_width/self_scale

            hight_ge = round( self.height*scale)
            wight_ge = round(self.width*scale)
            imgwidth = showwidth
            imgheight = int(src_height*scale)

            num = int(imgwidth / wight_ge) if int(imgwidth / wight_ge) == imgwidth / wight_ge else int(
                imgwidth / wight_ge) + 1

            temp_index = [t for t in range(imgwidth)]
            temp_index = np.array(temp_index)
            frame_ground = np.zeros((int(imgheight), showwidth))
            temp_h=self.last_dets[:,3]-self.last_dets[:,1]
            temp_h.sort()
            temp_h_thre=np.mean(temp_h[:-100])
            temp_flag=[]
            temp_lag_s=10
            temp_lag_s=3
            for i in range(len(self.last_dets)):
                det=self.last_dets[i][:4]
                det_cxy=xyxy2cxywh(det)
                t1=self.last_dets[:,0]>det_cxy[0]-temp_lag_s*det_cxy[2]
                t2=self.last_dets[:,1]>det_cxy[1]-temp_lag_s*det_cxy[3]
                t3 = self.last_dets[:, 0] < det_cxy[0] + temp_lag_s * det_cxy[2]
                t4 = self.last_dets[:, 1] < det_cxy[1] + temp_lag_s * det_cxy[3]
                t1[i]=False
                temp_t=[h1 and h2 and h3 and h4 for h1,h2,h3,h4 in zip(t1,t2,t3,t4)]
                temp_det=self.last_dets[temp_t]
                temp_h=(temp_det[:,3]-temp_det[:,1])*(temp_det[:,2]-temp_det[:,0])
                temp_sum_flag=temp_h<det_cxy[3]
                if det_cxy[2]*det_cxy[3]/np.mean(temp_h)>temp_lag_s or min(self.last_dets[i][4],self.last_dets[i][5])<0.5:
                    temp_flag.append(False)
                else:
                    temp_flag.append(True)

                temp_h.sort()


            self.last_dets=self.last_dets[temp_flag]
            for i in range(len(self.last_dets)) :
                det=self.last_dets[i]
                bbox=det[:4].copy()
                bbox*=scale
                bbox=np.array(bbox,dtype=int)
                frame_ground[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 2

            while end_xy[0] < num:
                count_w += 1
                all_xy.append(end_xy)
                end_xy, frame_ground, temp_label, temp_bbox, temp_min_ge = self.get_xy(end_xy, frame_ground, num, self.last_dets.copy(),scale,temp_min_ge)
                all_label.append(temp_label)
                all_xy2.append(end_xy)
                all_bbox.append(temp_bbox)
                min_h = self.get_h(frame_ground, num, hight_ge)
                gminh = int(min_h * hight_ge)
                if gminh >= imgheight and min_h >= num:
                    break

                temp_1 = frame_ground[gminh, :] == 0
                temp_2 = frame_ground[gminh, :] == 2
                temp_3 = [temp_1[t] or temp_2[t] for t in range(len(temp_1))]


                while (True in temp_3)==False:
                    min_h += 1
                    gminh = round(min_h * round(self.height * scale))
                    if gminh >= imgheight and min_h >= num:
                        break
                    temp_1 = frame_ground[gminh, :] == 0
                    temp_2 = frame_ground[gminh, :] == 2
                    temp_3 = [temp_1[t] or temp_2[t] for t in range(len(temp_1))]
                if gminh >= imgheight and min_h >= num:
                    break
                end_xy = [round(temp_index[temp_3][0] / round(self.width * scale)), min_h]


            temp_scales=[1,1.2]
            temp_scales=[1.2]
            for self_scale2 in temp_scales:
                hight_ge = self.height
                wight_ge = self.width
                hight_ge2 = self.height*self_scale2
                wight_ge2 = self.width*self_scale2
                for i in range(len(all_xy)):
                    temp_xy1 = all_xy[i]
                    temp_xy2 = all_xy2[i]
                    temp_xy1 = [int(temp_xy1[0] * wight_ge), int(temp_xy1[1] * hight_ge)]
                    temp_xy2 = [int(temp_xy2[0] * wight_ge2), int(temp_xy2[1] * hight_ge2)]

                    img0 = src_img[temp_xy1[1]:temp_xy2[1], temp_xy1[0]:temp_xy2[0]]
                    tic=time.time()
                    try:
                        img, ratio, dw, dh = letterbox(img0, height=self.height, width=self.width)
                    except:
                        continue
                    toc=time.time()
                    loc_info = (self_scale, temp_xy1[0], temp_xy1[1], ratio, dw, dh)
                    all_temp_count += 1
                    img = img.transpose(2, 0, 1)  # BGR
                    img = np.ascontiguousarray(img, dtype=np.float32)
                    sub_img_list.append((img0, img, loc_info))

        return sub_img_list


    def _sliding_window_orgin(self, src_giga_img):
        sub_img_list = []
        raw_height, raw_width = src_giga_img.shape[:2]

        temp_scales=(0.2,0.6)
        for scale in temp_scales:#multi-scale;
            n = 0
            src_img = cv2.resize(src_giga_img, (int(raw_width * scale), int(raw_height * scale)))
            src_height, src_width = src_img.shape[:2]
            # sub image generate
            ini_y = 0
            while ini_y < src_height:
                ini_x = 0
                while ini_x < src_width:
                    img0 = src_img[ini_y:ini_y + self.height, ini_x:ini_x + self.width]
                    #cv2.imwrite("img0.jpg", img0)
                    loc_info = (scale, ini_x, ini_y)
                    # Padded resize
                    img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
                    img = img.transpose(2, 0, 1)#BGR
                    img = np.ascontiguousarray(img, dtype=np.float32)

                    sub_img_list.append((img0, img, loc_info))

                    n += 1
                    if ini_x == src_width - self.width:
                        break
                    ini_x += self.x_step
                    if ini_x + self.width > src_width:
                        ini_x = src_width - self.width
                if ini_y == src_height - self.height:
                    break
                ini_y += self.y_step
                if ini_y + self.height > src_height:
                    ini_y = src_height - self.height
        return sub_img_list

def show_img(img,bbox):
    imgheight, imgwidth = img.shape[:2]
    showwidth = 1920
    scale = showwidth / imgwidth
    img = cv2.resize(img, (int(imgwidth * scale), int(imgheight * scale)))

    bbox*=scale
    bbox=np.array(bbox,dtype=int)
    for box in bbox:
        cv2.rectangle(img, box[0:2], box[2:4], color=(0,0,255), thickness=2)

    return img
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="/data/PANDA/PANDA-Image/image_test/14_OCT_Habour/1", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to show the inference result of image/video",
    )
    parser.add_argument(
        "--show_det",
        action="store_true",
        help="whether to show the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    parser.add_argument("--conf", default=0.15, type=float, help="test conf")
    parser.add_argument("--nms", default=0.53, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=10, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )


    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img, ratio = preproc_new(img, self.rgb_means, self.std)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info



@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def fast_nms(boxes, scores, iou_thres: float = 0.5, top_k: int = 200, second_threshold: bool = False,
             conf_thres: float = 0.5):
    '''
    Vectorized, approximated, fast NMS, adopted from YOLACT:
    https://github.com/dbolya/yolact/blob/master/layers/functions/detection.py
    The original version is for multi-class NMS, here we simplify the code for single-class NMS
    '''
    scores, idx = scores.sort(0, descending=True)

    idx = idx[:top_k].contiguous()
    scores = scores[:top_k]
    num_dets = idx.size()

    boxes = boxes[idx, :]

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=0)

    keep = (iou_max <= iou_thres)

    if second_threshold:
        keep *= (scores > conf_thres)

    return idx[keep]


def draw_rect(img,bboxs):
    bboxs=bboxs.data.cpu().numpy()
    bboxs=np.array(bboxs,dtype=int)
    for bbox in bboxs:
        cv2.rectangle(img, bbox[:2], bbox[2:], (0, 0, 255), 3)
    return img

def image_demo(predictor,  current_time, args):
    name_all = (args.path).strip().split('/')
    file_name = name_all[-2] + ".txt"
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    name_all = (args.path).strip().split('/')
    file_name = name_all[-1] + ".txt"
    args.obj = name_all[-1]
    dataloader = LoadPandaImages(args.path,args, exp.test_size, predictor)  # test_size is the the size of image block

    timer = Timer()
    timer1 = Timer()
    timer2 = Timer()
    frame_id = 1
    results = []

    for path, src_giga_img, sub_img_list in dataloader:
        img_info = {}
        img_info['height'] = src_giga_img.shape[0]
        img_info['width'] = src_giga_img.shape[1]
        clone_flag = False
        pred = torch.Tensor()
        tic = time.time()
        tic_det = time.time()
        timer.tic()
        temp_results = []
        im_blob_array = []
        cat_count = 0
        for img_tuple in sub_img_list:
            tic1 = time.time()
            img02, img, loc_info = img_tuple
            im_blob = img.transpose(1, 2, 0)
            outputs, img_info = predictor.inference(im_blob, timer1)
            timer1.toc()
            toc1 = time.time()
            if outputs[0] is not None:
                disp = torch.zeros(outputs[0].size(), device='cuda:0')
                disp[:, 0], disp[:, 1] = loc_info[1:3]
                disp[:, 2], disp[:, 3] = loc_info[1:3]
                temp_a_num = img02.shape[1] / (exp.test_size[1] / loc_info[3])
                # new_scale=img02.shape[1]/exp.test_size[1]
                new_scale = img_info['ratio'] / loc_info[3]
                # new_scale2=img02.shape[0]/exp.test_size[0]
                new_scale2 = img_info['ratio'] / loc_info[3]
                outputs[0][:, 0] = (outputs[0][:, 0] - loc_info[4]) * new_scale  #####################
                outputs[0][:, 2] = (outputs[0][:, 2] - loc_info[4]) * new_scale  #####################
                outputs[0][:, 1] = (outputs[0][:, 1] - loc_info[5]) * new_scale2  #####################
                outputs[0][:, 3] = (outputs[0][:, 3] - loc_info[5]) * new_scale2  #####################
                pred_sub = torch.add(outputs[0], disp)
                pred_sub[:, :4] = pred_sub[:, :4] / loc_info[0]
                if not clone_flag:
                    clone_flag = True
                    pred = pred_sub.clone()
                else:
                    pred = torch.cat((pred, pred_sub), 0)
            toc2 = time.time()
            toc = time.time()
            timer.toc()
            toc_det = time.time()
            if len(pred) > 0:
                tic = time.time()
                try:
                    dets = non_max_suppression(pred.unsqueeze(0),args.conf,args.nms)[0].cpu()
                except:
                    pass
                toc = time.time()
            else:
                # online_im = img_info['raw_img']
                online_im = src_giga_img
                continue
            outputs = dets.clone()
        toc_det = time.time()
        print('3    detection object cost time: ',toc_det-tic_det)
        # if outputs[0] is not None:
        temp_outputs = []
        outputs = np.array(outputs)
        temp_results=[]
        if outputs is not None:
            temp_outputs = outputs
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        if args.save_result:
            temp_save_txt_det = '{}/{}'.format(args.output_dir_txt_det, path.split('/')[-2])
            os.makedirs(temp_save_txt_det, exist_ok=True)
            res_file = osp.join(temp_save_txt_det, path.split('/')[-1].replace('.jpg', '.txt'))
            np.savetxt(res_file, temp_outputs, delimiter=',', fmt='%.2f')
        if args.show_det:
            img=show_img(src_giga_img,temp_outputs)
            cv2.imshow('demo', img)
            show_time=1
            key = cv2.waitKey(show_time) & 0xFF
            if key == ord(' '):
                cv2.waitKey(0)
            if key == ord('q'):
                break
        frame_id = frame_id + 1
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    out_ckpt = args.ckpt.split('.')[0].split('/')[-1]
    temp_out_name = 'spdet'
    output_dir_txt_det = osp.join(exp.output_dir, temp_out_name,out_ckpt,
                                  args.experiment_name)
    os.makedirs(output_dir_txt_det, exist_ok=True)
    args.output_dir_txt_det = output_dir_txt_det

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict

        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    image_demo(predictor, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
