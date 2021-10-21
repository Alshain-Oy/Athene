#!/usr/bin/env python3

import sys
import cv2
import time
import numpy as np
import ctypes

import libAthene


# Make all windows map pixels 1:1 to screen resolution
awareness = ctypes.c_int()
errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)



img = cv2.imread( sys.argv[2] )
template = cv2.imread( sys.argv[1] )


matcher = libAthene.ShapeMatcher()

matcher.configure(imageScale = 0.5, useUnsharpMask=True)

matcher.add_template( "target", template )

t0 = time.time()
results, image = matcher.detect( "target", img )

t1 = time.time()
print( "matcher.detect took: %.1f ms" %((t1-t0)*1000))

import pprint
pprint.pprint( results )


h,w = template.shape[:2]

cv2.line( template, np.int0((w/2 - 5, h/2)), np.int0((w/2+5, h/2)), (255,0,0), 2)
cv2.line( template, np.int0((w/2, h/2-5)), np.int0((w/2, h/2+5)), (255,0,0), 2)



cv2.imshow( "Image", image)
cv2.imshow( "Template", template)

cv2.waitKey(-1)