'''
@Author: JosieHong
@Date: 2020-05-06 12:57:47
@LastEditAuthor: JosieHong
@LastEditTime: 2020-06-25 00:01:23
'''
from .davis_eval import DAVISeval

def davis_eval(result_files, davis):
	davis_dets = davis.loadRes(result_files['segm'])
	davisEval = DAVISeval(davis, davis_dets)
	davisEval.evaluate()