import matplotlib.pyplot as plt
import gluoncv
import cv2
import os
import numpy as np
import mxnet as mx
from ensemble_boxes import *
from gluoncv import model_zoo, data, utils
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import supervision as sv
import streamlit as st

def load_models():
    """
    Loading all models

    Parameters
    ----------
        None
    returns
    -------
        mask_generator: SamPredictor
            SAM model object
        yolo: YOLO
            YOLO model object from model_zoo
        frcnn: FRCNN
            FRCNN model object from model_zoo
    """
    sam = sam_model_registry["vit_b"](checkpoint=r"C:\Users\roy.smith\Downloads\sam_vit_b_01ec64.pth")
    mask_generator = SamPredictor(sam)
    yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    frcnn = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    return mask_generator, yolo, frcnn

def get_predictions(file_path, yolo_model, frcnn_model):
    """
    Detect objects for a given image using YOLO and FRCNN model

    Parameters
    ----------
        file_path : string
            Database server IP
        yolo_model : YOLO
            YOLO model object
        frcnn_model : FRCNN
            FRCNN model object
    returns
    -------
        yolo_ids: mxnet.ndarray
            Object IDs predicted by YOLO model
        yolo_scores: mxnet.ndarray
            Prediction probability for each object predicted by YOLO model
        yolo_bbx: mxnet.ndarray
            Boundary box coordinates for each object predicted by YOLO model
        frcnn_ids: mxnet.ndarray
            Object IDs predicted by FRCNN model
        frcnn_scores: mxnet.ndarray
            Prediction probability for each object predicted by FRCNN model
        frcnn_bbx: mxnet.ndarray
            Boundary box coordinates for each object predicted by FRCNN model
    """
    x, img = data.transforms.presets.yolo.load_test(file_path, short=512)
    yolo_ids, yolo_scores, yolo_bbx = yolo_model(x)
    x, img = data.transforms.presets.rcnn.load_test(file_path, short=512)
    frcnn_ids, frcnn_scores, frcnn_bbx = frcnn_model(x)
    return yolo_ids, yolo_scores, yolo_bbx, frcnn_ids, frcnn_scores, frcnn_bbx, img

def get_ensemble_prediction(yolo_ids, yolo_scores, yolo_bbx, frcnn_ids, frcnn_scores, frcnn_bbx):
    """
    Generate an ensemble prediction using objects predicted by YOLO and FRCNN model

    Parameters
    ----------
        yolo_ids: mxnet.ndarray
            Object IDs predicted by YOLO model
        yolo_scores: mxnet.ndarray
            Prediction probability for each object predicted by YOLO model
        yolo_bbx: mxnet.ndarray
            Boundary box coordinates for each object predicted by YOLO model
        frcnn_ids: mxnet.ndarray
            Object IDs predicted by FRCNN model
        frcnn_scores: mxnet.ndarray
            Prediction probability for each object predicted by FRCNN model
        frcnn_bbx: mxnet.ndarray
            Boundary box coordinates for each object predicted by FRCNN model
    returns
    -------
        boxes: np.ndarray
            Final boundary boxes predicted by the ensemble model
        scores: np.ndarray
            Prediction predicted by the ensemble model
        labels: np.ndarray
            Labels for objects predicted by the ensemble model
        bbx_max: float
            Object IDs predicted by FRCNN model
        bbx_min: float
            Prediction probability for each object predicted by FRCNN model
    """
    bbx1 = yolo_bbx[0].asnumpy()
    bbx2 = frcnn_bbx[0].asnumpy()
    bbx1 = bbx1[~np.all(bbx1 == -1, axis=1)]
    bbx2 = bbx2[~np.all(bbx2 == -1, axis=1)]
    bbx = [bbx1.tolist(), bbx2[:bbx1.shape[0]].tolist()]
    bbx_max, bbx_min = np.max(bbx), np.min(bbx)
    bbx = (bbx-bbx_min)/(bbx_max-bbx_min)

    class1 = yolo_ids[0].asnumpy()
    class2 = frcnn_ids[0].asnumpy()
    class1 = class1[~np.all(class1 == -1, axis=1)]
    class2 = class2[~np.all(class2 == -1, axis=1)]
    classes = [class1.flatten(), class2[:class1.shape[0]].flatten()]

    scores1 = yolo_scores[0].asnumpy()
    scores2 = frcnn_scores[0].asnumpy()
    scores1 = scores1[~np.all(scores1 == -1, axis=1)]
    scores2 = scores2[~np.all(scores2 == -1, axis=1)]
    scores = [scores1.flatten(), scores2[:scores1.shape[0]].flatten()]

    boxes, scores, labels = weighted_boxes_fusion(bbx,
                                                  scores,
                                                  classes,
                                                  weights=[2, 1], iou_thr=0.5, skip_box_thr=0.0001)
    return boxes, scores, labels, bbx_max, bbx_min

def generate_mask(original_img, b_boxes, bbx_max, bbx_min):
    """
    Generate object mask for each dected object based on the boundary box predicted by ensemble model

    Parameters
    ----------
        original_img: mxnet.ndarray
            Object IDs predicted by YOLO model
        b_boxes: mxnet.ndarray
            Prediction probability for each object predicted by YOLO model
        bbx_max: float
            Maximum value of the boundary box normalization
        bbx_min: float
            Minimum value of the boundary box normalization
    returns
    -------
        final_img: np.ndarray
            Final image with boundary boxes and object mask
    """
    image_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    mask_generator.set_image(image_rgb)
    boxes = (b_boxes*(bbx_max-bbx_min)+bbx_min)
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
    final_img = original_img.copy()
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    for box in boxes:
        mask_generator.set_image(final_img)
        masks, scores_, logits = mask_generator.predict(
            box=box,
            multimask_output=True
        )
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]
        source_image = box_annotator.annotate(scene=final_img.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=source_image.copy(), detections=detections)
        final_img = segmented_image
    return final_img


st.title("Object detection")
col1, col2 = st.columns(2)
up_fileobj = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if up_fileobj is not None:
    with open(os.path.join(up_fileobj.name), "wb") as f:
        f.write((up_fileobj).getbuffer())
    up_imagebyte = model_ouptut = up_fileobj.getvalue()
    with st.spinner("Please wait.."):
        mask_generator, yolo, frcnn = load_models()
        yolo_ids, yolo_scores, yolo_bbx, frcnn_ids, \
            frcnn_scores, frcnn_bbx, original_img = get_predictions(up_fileobj.name, yolo, frcnn)
        boxes, scores, labels, bbx_max, bbx_min = get_ensemble_prediction(yolo_ids, yolo_scores, yolo_bbx,
                                                                          frcnn_ids, frcnn_scores, frcnn_bbx)
        model_ouptut = generate_mask(original_img, boxes, bbx_max, bbx_min)
    col1.image(up_imagebyte)
    col2.image(model_ouptut)