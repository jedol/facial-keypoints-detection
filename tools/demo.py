import argparse
import numpy as np
import caffe
import cv2


def compute_keypoints(net, global_score, local_prob, local_score, params):
    ph,pw = local_prob.shape[1:]
    global_point = (np.clip(global_score*params['SHAPE_STD'], -1, 1)+\
                    params['SHAPE_MEAN'])*params['INPUT_SIZE']*0.5
    all_local_prob = local_prob.reshape(params['NUM_LANDMARK'],-1)
    bins_x, bins_y = np.meshgrid((np.arange(pw)+0.5)*params['FEATURE_STRIDE'],
                                 (np.arange(ph)+0.5)*params['FEATURE_STRIDE'])
    bins_origin = np.hstack((bins_x.reshape(-1,1), bins_y.reshape(-1,1)))-\
                  params['LOC_REGION_SIZE']/2
    local_delta = local_score.transpose(0,2,3,1).reshape(params['NUM_LANDMARK'],-1,2)
    local_delta = (np.clip(local_delta*params['LOC_POINT_STD'], -1, 1)+\
                   params['LOC_POINT_MEAN'])*params['LOC_REGION_SIZE']*0.5
    all_local_points = bins_origin+local_delta

    dist = np.linalg.norm(all_local_points-np.expand_dims(global_point, 1),
                          axis=2)/params['INPUT_SIZE']
    cost = all_local_prob+params['OPT_ALPHA']*np.exp(-dist)
    inds = cost.argmax(axis=1)

    local_points = all_local_points[np.arange(params['NUM_LANDMARK']), inds]
    prob = all_local_prob[np.arange(params['NUM_LANDMARK']), inds]

    return local_points*prob.reshape(-1,1)+global_point*(1-prob.reshape(-1,1))


if __name__=="__main__":
    ## parse arguments
    parser = argparse.ArgumentParser(description="Demo script for detecting facial keypoints")
    parser.add_argument("image", help="path to test image.")
    parser.add_argument("bbox", help="path to text file that contains bounding coordinate of face. (x1,y1,x2,y2)")
    parser.add_argument("netpt", help="path to prototext file that defines network.")
    parser.add_argument("caffemodel", help="path to trained caffemodel file.")

    args = parser.parse_args()
    image_file_path = args.image
    bbox_file_path = args.bbox
    deploy_netpt_path = args.netpt
    trained_model_path = args.caffemodel

    ## parameters
    params = dict()
    params['INPUT_SIZE'] = 160
    params['FEATURE_STRIDE'] = 8
    params['NUM_LANDMARK'] = 5
    params['SHAPE_MEAN'] = (1.0, 1.0)
    params['SHAPE_STD'] = (0.2, 0.2)
    params['LOC_REGION_SIZE'] = 14
    params['LOC_POINT_MEAN'] = (1.0, 1.0)
    params['LOC_POINT_STD'] = (0.2, 0.2)
    params['OPT_ALPHA'] = 4

    ## load image and bbox
    image = cv2.imread(image_file_path)
    bbox = np.loadtxt(bbox_file_path, np.int32)

    ## initialiize net
    print 'Initialize network...'
    net = caffe.Net(deploy_netpt_path, trained_model_path, caffe.TEST)

    ## prepare data
    print 'Prepare image...'
    data = image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    scale = (float(params['INPUT_SIZE'])/data.shape[0],
             float(params['INPUT_SIZE'])/data.shape[1])
    data = cv2.resize(data, (params['INPUT_SIZE'], params['INPUT_SIZE']))
    data = data.transpose(2,0,1).astype(np.float32)[None]

    ## compute keypoints
    print 'Compute keypoints...'
    net.blobs['data'].reshape(*data.shape)
    forward_kwargs = {'data':data}
    _ = net.forward(**forward_kwargs)
    global_score = net.blobs['shape_score'].data[0]
    local_prob = net.blobs['cls_prob'].data[0,1:]
    local_score = net.blobs['loc_score'].data[0]
    points = compute_keypoints(net, global_score, local_prob, local_score, params)
    points = points/scale+bbox[:2]

    ## disply result
    image = cv2.rectangle(image, tuple(bbox[:2].tolist()),
                          tuple(bbox[2:].tolist()), (0,255,0))
    for p in points:
        pt = tuple(p.astype(np.int32).tolist())
        image = cv2.circle(image, pt, 1, (0,255,0))
    cv2.namedWindow('result')
    cv2.imshow('result', image)
    cv2.waitKey()
    cv2.destroyAllWindows()





