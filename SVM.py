import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
import skimage

from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


src = r'C:\Users\juwon\PycharmProjects\COMPSYS306_Phase_2\dataset_xy'


def resize_all(src, pklname, include, width=150, height=None):
    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1}) traffic sign images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))  # [:,:,::-1]
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im.flatten())

        joblib.dump(data, 'SVM_test_data.joblib')

print(os.listdir(src))

base_name = 'traffic_signs'
width = 150
include = {'green_light', 'red_light', 'stop_sign', 'thirty_sign', 'road'}
resize_all(src=src, pklname=base_name, width=width, include=include)


data = joblib.load('SVM_test_data.joblib')

print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

print(Counter(data['label']))


X = np.array(data['data'])
y = np.array(data['label'])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

# class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
#     """
#     Convert an array of RGB images to grayscale
#     """
#
#     def __init__(self):
#         pass
#
#     def fit(self, X, y=None):
#         """returns itself"""
#         return self
#
#     def transform(self, X, y=None):
#         """perform the transformation and return an array"""
#         return np.array([skimage.color.rgb2gray(img) for img in X])
#
#
# class HogTransformer(BaseEstimator, TransformerMixin):
#     """
#     Expects an array of 2d arrays (1 channel images)
#     Calculates hog features for each img
#     """
#
#     def __init__(self, y=None, orientations=9,
#                  pixels_per_cell=(8, 8),
#                  cells_per_block=(3, 3), block_norm='L2-Hys'):
#         self.y = y
#         self.orientations = orientations
#         self.pixels_per_cell = pixels_per_cell
#         self.cells_per_block = cells_per_block
#         self.block_norm = block_norm
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#
#         def local_hog(X):
#             return hog(X,
#                        orientations=self.orientations,
#                        pixels_per_cell=self.pixels_per_cell,
#                        cells_per_block=self.cells_per_block,
#                        block_norm=self.block_norm)
#
#         try:  # parallel
#             return np.array([local_hog(img) for img in X])
#         except:
#             return np.array([local_hog(img) for img in X])


# create an instance of each transformer
# grayify = RGB2GrayTransformer()
# hogify = HogTransformer(
#     pixels_per_cell=(14, 14),
#     cells_per_block=(2, 2),
#     orientations=9,
#     block_norm='L2-Hys'
# )
scalify = StandardScaler()

# call fit_transform on each transform converting X_train step by step
# X_train_gray = grayify.fit_transform(X_train)
# X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train)

print(X_train_prepared.shape)

# X_test_gray = grayify.transform(X_test)
# X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test)


param_grid={'C':[0.1],'gamma':[0.0001],'kernel':['poly']}
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model=GridSearchCV(svc,param_grid)
model.fit(X_train_prepared,y_train)
print('The Model is trained well with the given images')
print("\n The best parameters :\n",model.best_params_)

joblib.dump(model, 'SVM_model.joblib')


y_pred = model.predict(X_test_prepared)
print(f"The models is {accuracy_score(y_pred,y_test)*100}% accurate")



