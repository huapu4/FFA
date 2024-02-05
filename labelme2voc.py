from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import imgviz
import numpy as np
import labelme
'''
offical code for labelme to voc-format dataset
'''

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", help="input annotated directory")
    parser.add_argument("--output", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true", default=False)
    args = parser.parse_args()

    if osp.exists(args.output):
        print("Output directory already exists:", args.output)
        sys.exit(1)
    os.makedirs(args.output)
    os.makedirs(osp.join(args.output, "JPEGImages"))
    os.makedirs(osp.join(args.output, "SegmentationClass"))
    os.makedirs(osp.join(args.output, "SegmentationClassPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output, "SegmentationClassVisualization")
        )
    os.makedirs(osp.join(args.output, "SegmentationObject"))
    os.makedirs(osp.join(args.output, "SegmentationObjectPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output, "SegmentationObjectVisualization")
        )
    print("Creating dataset:", args.output)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output, "JPEGImages", base + ".jpg")
        out_cls_file = osp.join(
            args.output, "SegmentationClass", base + ".npy"
        )
        out_clsp_file = osp.join(
            args.output, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_clsv_file = osp.join(
                args.output,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        out_ins_file = osp.join(
            args.output, "SegmentationObject", base + ".npy"
        )
        out_insp_file = osp.join(
            args.output, "SegmentationObjectPNG", base + ".png"
        )
        if not args.noviz:
            out_insv_file = osp.join(
                args.output,
                "SegmentationObjectVisualization",
                base + ".jpg",
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        label_list = [None]
        for item in label_file.shapes:
            if item['label'] =='NP':
                label_list.append(item)
            else:
                label_list[0] =item

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_list,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # class label
        labelme.utils.lblsave(out_clsp_file, cls)
        np.save(out_cls_file, cls)
        if not args.noviz:
            clsv = imgviz.label2rgb(
                label=cls,
                img=imgviz.rgb2gray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        # instance label
        labelme.utils.lblsave(out_insp_file, ins)
        np.save(out_ins_file, ins)
        if not args.noviz:
            instance_ids = np.unique(ins)
            instance_names = [str(i) for i in range(max(instance_ids) + 1)]
            insv = imgviz.label2rgb(
                label=ins,
                img=imgviz.rgb2gray(img),
                label_names=instance_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_insv_file, insv)


if __name__ == "__main__":
    main()
