'''
@Author: JosieHong
@Date: 2020-05-06 12:57:47
@LastEditAuthor: JosieHong
@LastEditTime: 2020-05-08 18:35:13
'''
from .davis_eval import DAVISeval

def davis_eval(result_file, davis):
	davis_dets = davis.loadRes(result_file)
	davisEval = DAVISeval(davis, davis_dets)
	davisEval.evaluate()