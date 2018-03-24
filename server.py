import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from sklearn.externals import joblib
clf = joblib.load('classifier.pkl')
scl = joblib.load('scaler.pkl')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            features = get_features(filepath)
            scaled_features = scale_features(features)
            print scaled_features
            pred, prob = classify_with_scaled_features(scaled_features, True)
            print pred
            print prob
            value = "falling" if ' '.join([str(j) for j in pred.tolist()])=="1" else "not falling"
            return value
    return 0


import numpy as np
import cv2

def get_CCA_from_image(image, threshold = 115, show_image=False):
    img = cv2.imread(image,0)
    h,w = img.shape[:2]


    ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    max_area_index=-1
    for i in range(len(stats)):
        if stats[i][0]==0 and stats[i][1]==0:
            continue
        if max_area_index==-1:
            max_area_index=i
        elif stats[i][4] > stats[max_area_index][4]:
            max_area_index=i

    chosen_box = thresh[stats[max_area_index][1]:stats[max_area_index][1]+stats[max_area_index][3],stats[max_area_index][0]:stats[max_area_index][0]+stats[max_area_index][2]]
    return stats[max_area_index], chosen_box, thresh, img


def get_best_threshold(img_filename, start=50, max_steps =100, alpha=1):
    min_ratio = float("inf")
    index = -1
    final_chosen_box = None
    final_stats = []
    final_img = None
    final_thresh = None
    thresholds = np.arange(start,start+max_steps,alpha)
    for i in thresholds:
        stats, chosen_box, thresh, img = get_CCA_from_image(img_filename, i, show_image=False)
        ratio = get_ratio_of_image(chosen_box)
        if ratio < min_ratio:
            min_ratio = ratio
            index = i
            final_chosen_box = chosen_box
            final_stats = stats
            final_thresh = thresh
            final_img = img
    print index
    return index, final_chosen_box, final_stats, final_thresh, final_img

def get_ratio_of_image(img):
    a=cv2.countNonZero(img)
    h,w = img.shape[:2]
    return float(float(a)/(float(h)*float(w)))

def get_features(filename, output=False):
    index, final_chosen_box, final_stats, final_thresh, final_img = get_best_threshold(filename)
    features = []
    area = final_stats[2]*final_stats[3]
    features.append(area)
    bb_ratio =max((float(final_stats[2])/float(final_stats[3])),(float(final_stats[3])/float(final_stats[2])))
    features.append(bb_ratio)
    hu_moments = cv2.HuMoments(cv2.moments(final_chosen_box))
    features.extend(hu_moments.T[0].tolist())
    variance = np.var(final_chosen_box)
    #features.append(variance)
    if output:
        print "threshold used: " + str(index)
        print "area: "+str(area)
        print "bb_ratio: "+ str(bb_ratio)
        print "hu_moments: "+np.array2string(hu_moments.T[0])
        print "variance: "+str(variance)
    return features

def scale_features(features):
    if not scl:
        return
    if len(features) != (scl.min_).shape[0]:
        return
    return scl.transform([features])

def classify_with_scaled_features(scaled_features, probabilities = False):
    if not clf:
        return
    if probabilities:
        return clf.predict(scaled_features), clf.predict_proba(scaled_features)
    return clf.predict(scaled_features)

