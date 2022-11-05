import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('KLCC', cv2.WINDOW_NORMAL)
cv2.imshow('KLCC',img)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

cm = plt.cm.get_cmap('RdYlBu_r')
n,bins,patches=plt.hist(img.ravel(),256,[0,256])
print(patches)
plt.title('Histogram of greyscale using OpenCV (hist) function')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.show()
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()