import cv2
def denoise(image_name, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # plt.subplot(212), plt.title("Original image"), plt.imshow(img)

    # plt.subplot(312), plt.imshow(gray)

    # thresholding
    thresh, thresh_img = cv2.threshold(img, 200, 255, 0, cv2.THRESH_BINARY)

    # erode the image to *enlarge* black blobs
    erode = cv2.erode(thresh_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

    # fill in the black blobs that are not surrounded by white:
    _, filled, _, _ = cv2.floodFill(erode, None, (0, 0), 255)

    # binary and with the threshold image to get rid of the thickness from erode
    out = (filled == 0) & (thresh_img == 0)
    # also

    out = cv2.bitwise_and(filled, thresh_img)
    file_name = image_name.split('/')[-1]
    save_name = f'Denoised_Images/denoised_{file_name}'

    # scale_percent = 10
    #
    # # calculate the 50 percent of original dimensions
    # width = int(out.shape[0] * scale_percent / 60)
    # height = int(out.shape[1] * scale_percent / 60)
    #
    # # dsize
    # dsize = (width, height)
    #
    # # resize image
    # output = cv2.resize(out, dsize)

    cv2.imwrite(save_name, out, params=None)

    return out
