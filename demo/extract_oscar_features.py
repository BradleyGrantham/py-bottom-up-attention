"""Extract the features needed to use Oscar."""
import base64
import json
import os
import errno

import click
import cv2
import numpy as np
import torch
import yaml

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.data import MetadataCatalog
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances


PACKAGE_LOCATION = os.path.dirname(os.path.dirname(__file__))
D2_ROOT = os.path.join(
    PACKAGE_LOCATION, "detectron2/model_zoo/"
)  # Root of detectron2
MIN_BOXES = 36
MAX_BOXES = 36


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image,
    cuda=True,
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    if cuda:
        torcharange = torch.arange(num_objs).cuda()
    else:
        torcharange = torch.arange(num_objs)

    idxs = torcharange * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep


def doit(detector, raw_images, cuda=True):
    with torch.no_grad():
        # Preprocessing
        inputs = []
        for raw_image in raw_images:
            image = detector.transform_gen.get_transform(
                raw_image
            ).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append(
                {
                    "image": image,
                    "height": raw_image.shape[0],
                    "width": raw_image.shape[1],
                }
            )
        images = detector.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(
            images, features, None
        )

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(
            dim=[2, 3]
        )  # (sum_proposals, 2048), pooled to 1x1

        # Predict classes and boxes for each proposal.
        (
            pred_class_logits,
            pred_proposal_deltas,
        ) = detector.model.roi_heads.box_predictor(feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        for probs, boxes, image_size in zip(
            probs_list, boxes_list, images.image_sizes
        ):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes,
                    probs,
                    image_size,
                    score_thresh=0.2,
                    nms_thresh=nms_thresh,
                    topk_per_image=MAX_BOXES,
                    cuda=cuda,
                )
                if len(ids) >= MIN_BOXES:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(
            rcnn_outputs.num_preds_per_image
        )  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
            instances_list, inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            # we pass the height and width in as 1 so we get proportional heights and widths rather than raw numbers
            raw_instances = detector_postprocess(instances, 1, 1)
            raw_instances_list.append(raw_instances)

        return raw_instances_list, roi_features_list


def load_vg_classes():
    # Load VG Classes
    data_path = os.path.join(PACKAGE_LOCATION, "demo/data/genome/1600-400-20")

    vg_classes = []
    with open(os.path.join(data_path, "objects_vocab.txt")) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

        MetadataCatalog.get("vg").thing_classes = vg_classes
    class_names = MetadataCatalog.get("vg").as_dict()["thing_classes"]

    return class_names


def dump_features_to_tsv(out_dir, dataset_name, detector, pathXid, cuda=True):
    img_paths, img_ids = zip(*pathXid)
    imgs = [cv2.imread(img_path) for img_path in img_paths]
    instances_list, features_list = doit(detector, imgs, cuda)

    class_names = load_vg_classes()

    brads_labels_list = list()
    brads_features_list = list()
    fake_captions = list()
    fake_images = list()

    for i, (img, image_id, instances, features) in enumerate(
        zip(imgs, img_ids, instances_list, features_list)
    ):

        d = dict()
        boxes = instances.pred_boxes.tensor.to("cpu").numpy()
        preds = instances.scores.to("cpu").numpy()
        clses = instances.pred_classes.to("cpu").numpy()
        num_bboxes = boxes.shape[0]
        boxes_with_height_and_width = np.concatenate(
            (
                boxes,
                (boxes[:, 2] - boxes[:, 0])[:, np.newaxis],
                (boxes[:, 3] - boxes[:, 1])[:, np.newaxis],
            ),
            axis=1,
        )

        resnet_embeddings = features.to("cpu").numpy()

        embeddings_for_oscar = np.concatenate(
            (resnet_embeddings, boxes_with_height_and_width), axis=1
        )
        d["num_boxes"] = num_bboxes
        d["features"] = base64.b64encode(
            embeddings_for_oscar.flatten()
        ).decode()
        brads_features_list.append((image_id, json.dumps(d)))

        brads_labels_list.append(
            (
                image_id,
                json.dumps(
                    [
                        {
                            "class": class_names[id_],
                            "rect": [float(a) for a in list(bbox)],
                            "conf": float(conf.item()),
                        }
                        for id_, conf, bbox in zip(clses, preds, boxes)
                    ]
                ),
            )
        )
        fake_captions.append(
            {
                "image_id": image_id,
                "id": i,
                "caption": "A photo that has no caption.",
            }
        )
        fake_images.append({"id": image_id, "file_name": image_id})

    feature_filename = f"{dataset_name}.feature.tsv"
    label_filename = f"{dataset_name}.label.tsv"
    caption_filename = f"{dataset_name}_caption.json"
    coco_format_caption_filename = f"{dataset_name}_caption_coco_format.json"

    mkdir(out_dir)

    tsv_writer(
        brads_features_list, os.path.join(f"{out_dir}", feature_filename)
    )
    tsv_writer(brads_labels_list, os.path.join(f"{out_dir}", label_filename))

    with open(os.path.join(out_dir, caption_filename), "w") as f:
        json.dump(fake_captions, f)

    with open(os.path.join(out_dir, coco_format_caption_filename), "w") as f:
        json.dump(
            {
                "annotations": fake_captions,
                "images": fake_images,
                "type": "captions",
                "info": "dummy",
                "licenses": "dummy",
            },
            f,
        )

    dataset_yaml = {
        "label": label_filename,
        "feature": feature_filename,
        "caption": caption_filename,
    }

    with open(os.path.join(out_dir, f"{dataset_name}.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f)


def mkdir(path):
    # if it is the current folder, skip.
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def tsv_writer(values, tsv_file_name, sep="\t"):
    mkdir(os.path.dirname(tsv_file_name))
    tsv_file_name_tmp = tsv_file_name + ".tmp"
    with open(tsv_file_name_tmp, "wb") as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = (
                sep.join(
                    map(
                        lambda v: v.decode() if type(v) == bytes else str(v),
                        value,
                    )
                )
                + "\n"
            )
            v = v.encode()
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)


def load_image_ids(img_root):
    """images in the same directory are in the same split"""
    paths_and_ids = []
    for name in os.listdir(img_root):
        idx = name.split(".")[0]
        paths_and_ids.append((os.path.join(img_root, name), idx))
    return paths_and_ids


def build_model(cuda=False):
    """Build model and load weights for vg only."""
    cfg = get_cfg()  # Renew the cfg file
    cfg.merge_from_file(
        os.path.join(
            D2_ROOT,
            "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml",
        )
    )
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    cfg.MODEL.WEIGHTS = (
        "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    )
    cfg.SOLVER.IMS_PER_BATCH = 4

    if not cuda:
        cfg.DEVICE = "cpu"
        cfg.MODEL.DEVICE = "cpu"
    detector = DefaultPredictor(cfg)
    return detector


@click.command()
@click.argument("image_dir")
@click.option("--dataset-name", default="custom")
@click.option("--output-dir", default="custom")
@click.option("--cuda/--no-cuda", default=True)
def main(image_dir, dataset_name, output_dir, cuda):
    paths_and_ids = load_image_ids(image_dir)  # Get paths and ids
    detector = build_model(cuda)
    dump_features_to_tsv(
        output_dir, dataset_name, detector, paths_and_ids, cuda
    )


if __name__ == "__main__":
    main()
