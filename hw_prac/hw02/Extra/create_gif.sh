#!bash
#
###########################################################
# Author: Bhishan Poudel
# Date  : Sep 29, 2017
# Topic : Create gif from png images
###########################################################
#
convert -loop 0 -delay 100 test_images/test*.png cost_history.gif
